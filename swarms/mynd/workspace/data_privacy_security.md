# MYND Data Privacy & Security Research

**Author**: Research Specialist
**Date**: 2026-01-04
**Version**: 1.0
**Status**: COMPREHENSIVE RESEARCH DOCUMENT

---

## Executive Summary

This document provides detailed technical guidance for implementing privacy and security in the MYND AI thought-capture application. Given MYND handles highly sensitive personal data (thoughts, goals, voice recordings, personal patterns), security is not optional - it is a core product differentiator.

**Key Findings**:
1. Apple CryptoKit provides robust AES-256-GCM encryption with Secure Enclave support
2. Local-first architecture significantly reduces attack surface
3. CloudKit private database with client-side encryption achieves near-zero-knowledge
4. BYOK (Bring Your Own Key) model aligns privacy with business model
5. GDPR compliance requires specific technical implementations

---

## 1. End-to-End Encryption Options

### 1.1 Apple CryptoKit Framework

CryptoKit is Apple's modern cryptography framework (iOS 13+, macOS 10.15+). It provides hardware-accelerated cryptographic operations with a Swift-native API.

**Recommended Implementation for MYND**:

```swift
import CryptoKit
import Foundation

// MARK: - Symmetric Encryption Service

final class EncryptionService {

    // MARK: - AES-256-GCM (Recommended for data at rest)

    /// Encrypt data using AES-256-GCM
    /// - Parameters:
    ///   - data: Plaintext data to encrypt
    ///   - key: 256-bit symmetric key
    /// - Returns: Combined nonce + ciphertext + tag
    func encrypt(_ data: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)

        // Combined representation includes nonce, ciphertext, and tag
        guard let combined = sealedBox.combined else {
            throw EncryptionError.sealingFailed
        }

        return combined
    }

    /// Decrypt AES-256-GCM encrypted data
    /// - Parameters:
    ///   - combined: Combined nonce + ciphertext + tag
    ///   - key: 256-bit symmetric key
    /// - Returns: Decrypted plaintext data
    func decrypt(_ combined: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: combined)
        return try AES.GCM.open(sealedBox, using: key)
    }

    // MARK: - Key Generation

    /// Generate a cryptographically secure 256-bit key
    func generateKey() -> SymmetricKey {
        SymmetricKey(size: .bits256)
    }

    /// Derive key from password using HKDF
    /// (Note: For password-based key derivation, prefer scrypt via CommonCrypto)
    func deriveKey(from masterKey: SymmetricKey, salt: Data, info: Data) -> SymmetricKey {
        let derivedKey = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: masterKey,
            salt: salt,
            info: info,
            outputByteCount: 32
        )
        return derivedKey
    }
}

// MARK: - Encryption Errors

enum EncryptionError: Error {
    case sealingFailed
    case invalidCiphertext
    case keyDerivationFailed
    case secureEnclaveUnavailable
}
```

### 1.2 AES-256-GCM vs ChaCha20-Poly1305

| Algorithm | Hardware Acceleration | Use Case | CryptoKit Support |
|-----------|----------------------|----------|-------------------|
| **AES-256-GCM** | Yes (Apple Silicon, Intel AES-NI) | Data at rest, large files | Native |
| **ChaCha20-Poly1305** | Software only on most devices | Streaming, battery-sensitive | Native |

**Recommendation for MYND**: Use AES-256-GCM for all encryption needs:
- Apple devices have hardware AES acceleration
- SwiftData/CloudKit data is primarily at-rest
- AES-GCM is the industry standard for authenticated encryption

```swift
// ChaCha20-Poly1305 alternative (if needed for streaming)
extension EncryptionService {

    func encryptWithChaCha(_ data: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try ChaChaPoly.seal(data, using: key)
        guard let combined = sealedBox.combined else {
            throw EncryptionError.sealingFailed
        }
        return combined
    }

    func decryptWithChaCha(_ combined: Data, using key: SymmetricKey) throws -> Data {
        let sealedBox = try ChaChaPoly.SealedBox(combined: combined)
        return try ChaChaPoly.open(sealedBox, using: key)
    }
}
```

### 1.3 Key Derivation Functions

**For Password-Based Keys (User-entered passphrase)**:

CryptoKit does not include PBKDF2 or Argon2 directly. Use CommonCrypto for PBKDF2:

```swift
import CommonCrypto

extension EncryptionService {

    /// Derive key from user password using PBKDF2-HMAC-SHA256
    /// - Parameters:
    ///   - password: User-provided password
    ///   - salt: Cryptographically random salt (store alongside encrypted data)
    ///   - iterations: PBKDF2 iteration count (minimum 600,000 for 2025)
    /// - Returns: 256-bit derived key
    func deriveKeyFromPassword(
        _ password: String,
        salt: Data,
        iterations: UInt32 = 600_000  // OWASP 2024 recommendation
    ) throws -> SymmetricKey {
        let passwordData = Data(password.utf8)
        var derivedKey = Data(count: 32)

        let status = derivedKey.withUnsafeMutableBytes { derivedKeyPtr in
            salt.withUnsafeBytes { saltPtr in
                passwordData.withUnsafeBytes { passwordPtr in
                    CCKeyDerivationPBKDF(
                        CCPBKDFAlgorithm(kCCPBKDF2),
                        passwordPtr.baseAddress?.assumingMemoryBound(to: Int8.self),
                        passwordData.count,
                        saltPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        salt.count,
                        CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                        iterations,
                        derivedKeyPtr.baseAddress?.assumingMemoryBound(to: UInt8.self),
                        32
                    )
                }
            }
        }

        guard status == kCCSuccess else {
            throw EncryptionError.keyDerivationFailed
        }

        return SymmetricKey(data: derivedKey)
    }

    /// Generate cryptographically secure salt
    func generateSalt(length: Int = 32) -> Data {
        var salt = Data(count: length)
        _ = salt.withUnsafeMutableBytes { ptr in
            SecRandomCopyBytes(kSecRandomDefault, length, ptr.baseAddress!)
        }
        return salt
    }
}
```

**Argon2 Alternative**:

For stronger password-based key derivation, consider using a third-party Argon2 library:

```swift
// Using swift-argon2 package
// Package: https://github.com/nicklockwood/Argon2
import Argon2

extension EncryptionService {

    func deriveKeyWithArgon2(
        _ password: String,
        salt: Data
    ) throws -> SymmetricKey {
        let hash = try Argon2.hash(
            password: password,
            salt: salt,
            iterations: 3,
            memory: 65536,  // 64MB
            parallelism: 4,
            hashLength: 32,
            type: .id  // Argon2id - recommended for password hashing
        )
        return SymmetricKey(data: hash)
    }
}
```

### 1.4 Hardware-Backed Keys (Secure Enclave)

The Secure Enclave is a dedicated security coprocessor in Apple devices that provides hardware-level key protection. Keys stored in the Secure Enclave:
- Never leave the hardware
- Cannot be extracted, even by the operating system
- Require biometric or passcode authentication for use

```swift
import CryptoKit
import LocalAuthentication

// MARK: - Secure Enclave Key Manager

final class SecureEnclaveKeyManager {

    /// Create a new private key in the Secure Enclave
    /// - Parameter tag: Unique identifier for the key
    /// - Returns: The public key (private key remains in Secure Enclave)
    func createSecureEnclaveKey(tag: String) throws -> P256.Signing.PublicKey {
        // Check Secure Enclave availability
        guard SecureEnclave.isAvailable else {
            throw EncryptionError.secureEnclaveUnavailable
        }

        // Create key with authentication requirement
        let accessControl = SecAccessControlCreateWithFlags(
            kCFAllocatorDefault,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            [.privateKeyUsage, .biometryCurrentSet],  // Requires biometric
            nil
        )!

        let privateKey = try SecureEnclave.P256.Signing.PrivateKey(
            compactRepresentable: true,
            accessControl: accessControl
        )

        // Store key reference in Keychain
        try storeKeyReference(privateKey, tag: tag)

        return privateKey.publicKey
    }

    /// Sign data using Secure Enclave private key
    /// - Parameters:
    ///   - data: Data to sign
    ///   - tag: Key identifier
    /// - Returns: Digital signature
    func sign(_ data: Data, withKeyTag tag: String) throws -> Data {
        let privateKey = try retrievePrivateKey(tag: tag)
        let signature = try privateKey.signature(for: data)
        return signature.rawRepresentation
    }

    /// Encrypt data using ECIES with Secure Enclave key
    /// Note: Secure Enclave keys are signing keys; for encryption,
    /// derive a symmetric key using ECDH
    func encryptWithSecureEnclaveKey(
        _ data: Data,
        recipientPublicKey: P256.KeyAgreement.PublicKey,
        tag: String
    ) throws -> Data {
        // Perform ECDH to derive shared secret
        let privateKey = try SecureEnclave.P256.KeyAgreement.PrivateKey(
            compactRepresentable: true
        )

        let sharedSecret = try privateKey.sharedSecretFromKeyAgreement(
            with: recipientPublicKey
        )

        // Derive symmetric key from shared secret
        let symmetricKey = sharedSecret.hkdfDerivedSymmetricKey(
            using: SHA256.self,
            salt: Data(),
            sharedInfo: "MYND-E2EE".data(using: .utf8)!,
            outputByteCount: 32
        )

        // Encrypt with derived key
        let sealedBox = try AES.GCM.seal(data, using: symmetricKey)

        // Return ephemeral public key + ciphertext
        var result = Data()
        result.append(privateKey.publicKey.rawRepresentation)
        result.append(sealedBox.combined!)

        return result
    }

    // MARK: - Private Helpers

    private func storeKeyReference(
        _ key: SecureEnclave.P256.Signing.PrivateKey,
        tag: String
    ) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: tag.data(using: .utf8)!,
            kSecValueRef as String: key.rawRepresentation as CFData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        SecItemDelete(query as CFDictionary)  // Remove existing
        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.storeFailed(status)
        }
    }

    private func retrievePrivateKey(tag: String) throws -> SecureEnclave.P256.Signing.PrivateKey {
        let query: [String: Any] = [
            kSecClass as String: kSecClassKey,
            kSecAttrApplicationTag as String: tag.data(using: .utf8)!,
            kSecReturnRef as String: true
        ]

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let keyData = result as? Data else {
            throw KeychainError.notFound
        }

        return try SecureEnclave.P256.Signing.PrivateKey(
            dataRepresentation: keyData
        )
    }
}
```

### 1.5 End-to-End Encryption for Cloud Sync

For true E2EE where the cloud provider cannot read user data:

```swift
// MARK: - E2EE Cloud Sync Manager

final class E2EECloudSyncManager {

    private let encryptionService: EncryptionService
    private let keyManager: SecureEnclaveKeyManager
    private let keychain: KeychainService

    /// Master key for encrypting all user data
    /// Derived from user password + device-specific salt
    private var masterKey: SymmetricKey?

    // MARK: - Key Management

    /// Initialize E2EE with user-provided password
    /// Creates or derives master key for all encryption operations
    func initialize(password: String) async throws {
        // Check if this is a new device (no existing salt)
        if let existingSalt = try? keychain.retrieve(key: "e2ee_salt") {
            // Derive key from existing salt
            masterKey = try encryptionService.deriveKeyFromPassword(
                password,
                salt: existingSalt
            )

            // Verify key is correct by decrypting test value
            guard try await verifyMasterKey() else {
                throw E2EEError.invalidPassword
            }
        } else {
            // New setup - generate salt and verification token
            let salt = encryptionService.generateSalt()
            masterKey = try encryptionService.deriveKeyFromPassword(
                password,
                salt: salt
            )

            // Store salt locally (this is safe - salt is not secret)
            try keychain.store(salt, key: "e2ee_salt")

            // Create verification token
            let verificationData = "MYND_VERIFY".data(using: .utf8)!
            let encryptedVerification = try encryptionService.encrypt(
                verificationData,
                using: masterKey!
            )
            try keychain.store(encryptedVerification, key: "e2ee_verify")
        }
    }

    // MARK: - Data Encryption

    /// Encrypt data before sending to CloudKit
    func encryptForSync(_ data: Data) throws -> Data {
        guard let key = masterKey else {
            throw E2EEError.notInitialized
        }
        return try encryptionService.encrypt(data, using: key)
    }

    /// Decrypt data received from CloudKit
    func decryptFromSync(_ encryptedData: Data) throws -> Data {
        guard let key = masterKey else {
            throw E2EEError.notInitialized
        }
        return try encryptionService.decrypt(encryptedData, using: key)
    }

    // MARK: - Multi-Device Key Sync

    /// Export encrypted key bundle for device pairing
    /// Uses recipient's public key for secure transfer
    func exportKeyBundle(
        for recipientPublicKey: P256.KeyAgreement.PublicKey
    ) throws -> Data {
        guard let key = masterKey else {
            throw E2EEError.notInitialized
        }

        // Perform ECDH to derive shared secret with recipient
        let ephemeralPrivateKey = P256.KeyAgreement.PrivateKey()
        let sharedSecret = try ephemeralPrivateKey.sharedSecretFromKeyAgreement(
            with: recipientPublicKey
        )

        // Derive transport key
        let transportKey = sharedSecret.hkdfDerivedSymmetricKey(
            using: SHA256.self,
            salt: Data(),
            sharedInfo: "MYND-KEY-SYNC".data(using: .utf8)!,
            outputByteCount: 32
        )

        // Encrypt master key with transport key
        let masterKeyData = key.withUnsafeBytes { Data($0) }
        let encryptedKey = try encryptionService.encrypt(masterKeyData, using: transportKey)

        // Bundle: ephemeral public key + encrypted master key
        var bundle = Data()
        bundle.append(ephemeralPrivateKey.publicKey.rawRepresentation)
        bundle.append(encryptedKey)

        return bundle
    }

    private func verifyMasterKey() async throws -> Bool {
        guard let key = masterKey,
              let encrypted = try? keychain.retrieve(key: "e2ee_verify") else {
            return false
        }

        do {
            let decrypted = try encryptionService.decrypt(encrypted, using: key)
            return String(data: decrypted, encoding: .utf8) == "MYND_VERIFY"
        } catch {
            return false
        }
    }
}

enum E2EEError: Error {
    case notInitialized
    case invalidPassword
    case keyExportFailed
    case keyImportFailed
}
```

---

## 2. Local-First Privacy Model

### 2.1 Architecture Principles

MYND should adopt a local-first architecture where:
1. All data is stored locally by default
2. Cloud sync is opt-in and encrypted
3. AI processing happens locally when possible
4. API calls minimize data exposure

```
+------------------------------------------------------------------+
|                    MYND LOCAL-FIRST ARCHITECTURE                   |
+------------------------------------------------------------------+
|                                                                    |
|  USER DATA FLOW:                                                   |
|                                                                    |
|  Voice Input → On-Device Transcription → Local SwiftData Store    |
|       ↓                                                           |
|  [Local Processing]                                                |
|       ↓                                                           |
|  Optional: Encrypted Sync to CloudKit                             |
|       ↓                                                           |
|  Optional: Anonymized API Call to Claude                          |
|                                                                    |
+------------------------------------------------------------------+
```

### 2.2 On-Device Processing Benefits

| Aspect | Local Processing | Cloud Processing |
|--------|------------------|------------------|
| **Latency** | <100ms | 500ms-3000ms |
| **Privacy** | Data never leaves device | Data transmitted to servers |
| **Offline** | Always available | Requires internet |
| **Cost** | $0 per operation | API costs accumulate |
| **Control** | User owns data completely | Provider has access |

### 2.3 Local LLM Options for Sensitive Data

For processing sensitive thoughts without cloud API calls:

```swift
import CoreML

// MARK: - Local AI Service

final class LocalAIService {

    // Option 1: Apple's On-Device ML
    // Using NaturalLanguage framework for basic NLP

    /// Classify thought intent locally
    func classifyIntent(_ text: String) async -> ThoughtIntent {
        let tagger = NLTagger(tagSchemes: [.sentimentScore, .lexicalClass])
        tagger.string = text

        // Analyze sentiment
        let sentiment = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)

        // Basic intent classification based on keywords
        let lowercased = text.lowercased()

        if lowercased.contains("want to") || lowercased.contains("goal") {
            return .goal
        } else if lowercased.contains("need to") || lowercased.contains("must") {
            return .action
        } else if lowercased.contains("worried") || lowercased.contains("anxious") {
            return .concern
        } else {
            return .thought
        }
    }

    // Option 2: On-Device LLM (iOS 18+ with Apple Intelligence)
    // Note: Requires enrollment in Apple's AI frameworks

    #if canImport(AppleIntelligence)
    func processWithAppleIntelligence(_ text: String) async throws -> String {
        let session = try await AITextSession()
        let response = try await session.respond(to: text)
        return response
    }
    #endif

    // Option 3: Local LLaMA/Mistral via llama.cpp
    // For complete offline capability

    private var localModel: LlamaModel?

    func loadLocalModel() async throws {
        // Load quantized model (e.g., Mistral-7B-Q4)
        // Requires ~4GB RAM, works on iPhone 14 Pro+
        let modelPath = Bundle.main.path(forResource: "mistral-7b-q4", ofType: "gguf")!
        localModel = try LlamaModel(path: modelPath)
    }

    func generateLocalResponse(_ prompt: String) async -> String {
        guard let model = localModel else {
            return "Local model not loaded"
        }

        return await model.generate(
            prompt: prompt,
            maxTokens: 256,
            temperature: 0.7
        )
    }
}

enum ThoughtIntent: String {
    case thought
    case goal
    case action
    case concern
    case question
}
```

### 2.4 Hybrid Approach: Local + Cloud

```swift
// MARK: - Hybrid Processing Manager

final class HybridProcessingManager {

    private let localAI: LocalAIService
    private let cloudAI: ClaudeClient
    private let privacySettings: PrivacySettings

    /// Process thought with appropriate backend based on sensitivity and settings
    func process(_ thought: String) async throws -> ProcessingResult {

        // Step 1: Always do local classification first
        let intent = await localAI.classifyIntent(thought)
        let sensitivity = analyzeSensitivity(thought)

        // Step 2: Decide processing path
        switch (privacySettings.processingMode, sensitivity) {

        case (.localOnly, _):
            // User chose local-only mode
            return try await processLocally(thought, intent: intent)

        case (.hybrid, .high):
            // Sensitive content - process locally
            return try await processLocally(thought, intent: intent)

        case (.hybrid, .medium):
            // Medium sensitivity - anonymize before cloud
            let anonymized = anonymize(thought)
            return try await processWithCloud(anonymized, originalIntent: intent)

        case (.hybrid, .low), (.cloudPreferred, _):
            // Low sensitivity or user prefers cloud
            return try await processWithCloud(thought, originalIntent: intent)

        case (.cloudPreferred, .high):
            // Even in cloud-preferred mode, protect high sensitivity
            let anonymized = anonymize(thought)
            return try await processWithCloud(anonymized, originalIntent: intent)
        }
    }

    private func analyzeSensitivity(_ text: String) -> SensitivityLevel {
        let patterns: [(pattern: String, level: SensitivityLevel)] = [
            // High sensitivity patterns
            ("password", .high),
            ("credit card", .high),
            ("ssn", .high),
            ("social security", .high),
            ("bank account", .high),
            ("medical", .high),
            ("diagnosis", .high),
            ("therapy", .high),
            ("abuse", .high),

            // Medium sensitivity patterns
            ("salary", .medium),
            ("relationship", .medium),
            ("family", .medium),
            ("work problem", .medium),
        ]

        let lowercased = text.lowercased()

        for (pattern, level) in patterns {
            if lowercased.contains(pattern) {
                return level
            }
        }

        return .low
    }

    private func anonymize(_ text: String) -> String {
        var anonymized = text

        // Replace names (simplified - use NER in production)
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.address.rawValue)

        // Replace emails
        anonymized = anonymized.replacingOccurrences(
            of: #"[\w.]+@[\w.]+"#,
            with: "[EMAIL]",
            options: .regularExpression
        )

        // Replace phone numbers
        anonymized = anonymized.replacingOccurrences(
            of: #"\d{3}[-.]?\d{3}[-.]?\d{4}"#,
            with: "[PHONE]",
            options: .regularExpression
        )

        // Replace currency amounts
        anonymized = anonymized.replacingOccurrences(
            of: #"\$[\d,]+\.?\d*"#,
            with: "[AMOUNT]",
            options: .regularExpression
        )

        return anonymized
    }
}

enum SensitivityLevel {
    case low
    case medium
    case high
}

enum ProcessingMode {
    case localOnly
    case hybrid
    case cloudPreferred
}
```

---

## 3. Keychain & Secure Storage

### 3.1 Keychain Services Implementation

```swift
import Security
import LocalAuthentication

// MARK: - Keychain Service

final class KeychainService {

    private let serviceName: String
    private let accessGroup: String?

    init(serviceName: String = "com.mynd.app", accessGroup: String? = nil) {
        self.serviceName = serviceName
        self.accessGroup = accessGroup
    }

    // MARK: - Store Operations

    /// Store sensitive data with specified protection level
    func store(
        _ data: Data,
        key: String,
        accessibility: KeychainAccessibility = .whenUnlockedThisDeviceOnly,
        requireBiometric: Bool = false
    ) throws {
        var query = baseQuery(for: key)

        // Set accessibility
        query[kSecAttrAccessible as String] = accessibility.secValue

        // Configure biometric requirement
        if requireBiometric {
            var error: Unmanaged<CFError>?
            guard let accessControl = SecAccessControlCreateWithFlags(
                kCFAllocatorDefault,
                accessibility.secValue as CFTypeRef,
                .biometryCurrentSet,
                &error
            ) else {
                throw KeychainError.accessControlCreationFailed
            }
            query[kSecAttrAccessControl as String] = accessControl
        }

        query[kSecValueData as String] = data

        // Delete existing item first
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.storeFailed(status)
        }
    }

    /// Retrieve data from keychain
    func retrieve(key: String) throws -> Data {
        var query = baseQuery(for: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data else {
            throw KeychainError.notFound
        }

        return data
    }

    /// Delete item from keychain
    func delete(key: String) throws {
        let query = baseQuery(for: key)
        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    // MARK: - Biometric-Protected Operations

    /// Retrieve with biometric authentication
    func retrieveWithBiometric(
        key: String,
        reason: String
    ) async throws -> Data {
        var query = baseQuery(for: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        // Create LAContext for biometric prompt
        let context = LAContext()
        context.localizedReason = reason

        query[kSecUseAuthenticationContext as String] = context

        return try await withCheckedThrowingContinuation { continuation in
            var result: AnyObject?
            let status = SecItemCopyMatching(query as CFDictionary, &result)

            if status == errSecSuccess, let data = result as? Data {
                continuation.resume(returning: data)
            } else if status == errSecUserCanceled {
                continuation.resume(throwing: KeychainError.biometricCanceled)
            } else {
                continuation.resume(throwing: KeychainError.notFound)
            }
        }
    }

    // MARK: - Private Helpers

    private func baseQuery(for key: String) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key
        ]

        if let accessGroup = accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }

        return query
    }
}

// MARK: - Keychain Accessibility

enum KeychainAccessibility {
    case whenUnlocked
    case whenUnlockedThisDeviceOnly
    case afterFirstUnlock
    case afterFirstUnlockThisDeviceOnly
    case whenPasscodeSetThisDeviceOnly

    var secValue: CFString {
        switch self {
        case .whenUnlocked:
            return kSecAttrAccessibleWhenUnlocked
        case .whenUnlockedThisDeviceOnly:
            return kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        case .afterFirstUnlock:
            return kSecAttrAccessibleAfterFirstUnlock
        case .afterFirstUnlockThisDeviceOnly:
            return kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        case .whenPasscodeSetThisDeviceOnly:
            return kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly
        }
    }
}

// MARK: - Keychain Errors

enum KeychainError: Error {
    case storeFailed(OSStatus)
    case notFound
    case deleteFailed(OSStatus)
    case accessControlCreationFailed
    case biometricCanceled
    case biometricNotAvailable
}
```

### 3.2 Biometric Authentication (Face ID / Touch ID)

```swift
import LocalAuthentication

// MARK: - Biometric Authentication Service

final class BiometricService {

    private let context = LAContext()

    /// Check if biometric authentication is available
    var isBiometricAvailable: Bool {
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
    }

    /// Type of biometric available
    var biometricType: BiometricType {
        guard isBiometricAvailable else { return .none }

        switch context.biometryType {
        case .faceID:
            return .faceID
        case .touchID:
            return .touchID
        case .opticID:
            return .opticID  // Vision Pro
        @unknown default:
            return .none
        }
    }

    /// Authenticate user with biometrics
    func authenticate(reason: String) async throws -> Bool {
        guard isBiometricAvailable else {
            throw BiometricError.notAvailable
        }

        do {
            let success = try await context.evaluatePolicy(
                .deviceOwnerAuthenticationWithBiometrics,
                localizedReason: reason
            )
            return success
        } catch let error as LAError {
            switch error.code {
            case .userCancel:
                throw BiometricError.canceled
            case .biometryNotAvailable:
                throw BiometricError.notAvailable
            case .biometryNotEnrolled:
                throw BiometricError.notEnrolled
            case .biometryLockout:
                throw BiometricError.lockedOut
            default:
                throw BiometricError.failed(error)
            }
        }
    }

    /// Authenticate with fallback to passcode
    func authenticateWithFallback(reason: String) async throws -> Bool {
        do {
            let success = try await context.evaluatePolicy(
                .deviceOwnerAuthentication,  // Allows passcode fallback
                localizedReason: reason
            )
            return success
        } catch let error as LAError {
            throw BiometricError.failed(error)
        }
    }
}

enum BiometricType {
    case none
    case touchID
    case faceID
    case opticID  // Apple Vision Pro
}

enum BiometricError: Error {
    case notAvailable
    case notEnrolled
    case canceled
    case lockedOut
    case failed(LAError)
}
```

### 3.3 Data Protection API Levels

iOS provides file-level encryption through Data Protection:

```swift
import Foundation

// MARK: - Data Protection Service

final class DataProtectionService {

    /// Write data with specified protection level
    func writeProtectedData(
        _ data: Data,
        to url: URL,
        protection: FileProtection
    ) throws {
        try data.write(
            to: url,
            options: [.atomic, protection.writeOption]
        )
    }

    /// Set protection level on existing file
    func setProtection(_ protection: FileProtection, on url: URL) throws {
        try FileManager.default.setAttributes(
            [.protectionKey: protection.fileAttributeValue],
            ofItemAtPath: url.path
        )
    }

    /// Get current protection level of file
    func getProtection(of url: URL) throws -> FileProtection {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let value = attributes[.protectionKey] as? FileProtectionType else {
            return .completeUntilFirstUserAuthentication  // Default
        }
        return FileProtection(from: value)
    }
}

enum FileProtection {
    /// File is only accessible when device is unlocked
    /// Recommended for: Active data, thoughts being edited
    case complete

    /// File is accessible after first unlock until restart
    /// Recommended for: Background sync data
    case completeUntilFirstUserAuthentication

    /// File is only accessible when device is unlocked, but can be created while locked
    /// Recommended for: New captured thoughts
    case completeUnlessOpen

    /// No protection (not recommended)
    case none

    var writeOption: Data.WritingOptions {
        switch self {
        case .complete:
            return .completeFileProtection
        case .completeUntilFirstUserAuthentication:
            return .completeFileProtectionUntilFirstUserAuthentication
        case .completeUnlessOpen:
            return .completeFileProtectionUnlessOpen
        case .none:
            return []
        }
    }

    var fileAttributeValue: FileProtectionType {
        switch self {
        case .complete:
            return .complete
        case .completeUntilFirstUserAuthentication:
            return .completeUntilFirstUserAuthentication
        case .completeUnlessOpen:
            return .completeUnlessOpen
        case .none:
            return .none
        }
    }

    init(from type: FileProtectionType) {
        switch type {
        case .complete:
            self = .complete
        case .completeUnlessOpen:
            self = .completeUnlessOpen
        case .none:
            self = .none
        default:
            self = .completeUntilFirstUserAuthentication
        }
    }
}
```

### 3.4 Secure File Containers

For highly sensitive data, create an encrypted container:

```swift
// MARK: - Encrypted Container

final class EncryptedContainer {

    private let containerURL: URL
    private let encryptionService: EncryptionService
    private var encryptionKey: SymmetricKey?

    init(name: String) {
        let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!

        containerURL = documentsPath.appendingPathComponent(
            "\(name).encrypted",
            isDirectory: true
        )

        encryptionService = EncryptionService()
    }

    /// Unlock container with user-provided key
    func unlock(with key: SymmetricKey) throws {
        self.encryptionKey = key

        // Create container directory if needed
        if !FileManager.default.fileExists(atPath: containerURL.path) {
            try FileManager.default.createDirectory(
                at: containerURL,
                withIntermediateDirectories: true
            )

            // Set strictest protection
            try FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.complete],
                ofItemAtPath: containerURL.path
            )
        }
    }

    /// Lock container (clear key from memory)
    func lock() {
        encryptionKey = nil
    }

    /// Write encrypted file to container
    func writeFile(_ data: Data, name: String) throws {
        guard let key = encryptionKey else {
            throw ContainerError.locked
        }

        let encrypted = try encryptionService.encrypt(data, using: key)
        let fileURL = containerURL.appendingPathComponent(name)
        try encrypted.write(to: fileURL, options: .completeFileProtection)
    }

    /// Read and decrypt file from container
    func readFile(name: String) throws -> Data {
        guard let key = encryptionKey else {
            throw ContainerError.locked
        }

        let fileURL = containerURL.appendingPathComponent(name)
        let encrypted = try Data(contentsOf: fileURL)
        return try encryptionService.decrypt(encrypted, using: key)
    }

    /// List files in container
    func listFiles() throws -> [String] {
        guard encryptionKey != nil else {
            throw ContainerError.locked
        }

        return try FileManager.default.contentsOfDirectory(atPath: containerURL.path)
    }

    /// Securely delete entire container
    func destroy() throws {
        lock()
        try FileManager.default.removeItem(at: containerURL)
    }
}

enum ContainerError: Error {
    case locked
    case fileNotFound
    case encryptionFailed
}
```

---

## 4. Cloud Sync Security

### 4.1 CloudKit Encryption Architecture

CloudKit private database provides encryption at rest, but the data is accessible to Apple. For true privacy, implement client-side encryption:

```swift
import CloudKit

// MARK: - Secure CloudKit Sync Manager

final class SecureCloudKitManager {

    private let container: CKContainer
    private let privateDatabase: CKDatabase
    private let encryptionService: EncryptionService
    private var syncKey: SymmetricKey?

    init(containerIdentifier: String) {
        container = CKContainer(identifier: containerIdentifier)
        privateDatabase = container.privateCloudDatabase
        encryptionService = EncryptionService()
    }

    /// Initialize with user's sync key
    func configure(syncKey: SymmetricKey) {
        self.syncKey = syncKey
    }

    // MARK: - Encrypted Record Operations

    /// Save thought with client-side encryption
    func saveThought(_ thought: ThoughtNode) async throws {
        guard let key = syncKey else {
            throw CloudSyncError.notConfigured
        }

        // Serialize thought to JSON
        let encoder = JSONEncoder()
        let thoughtData = try encoder.encode(thought)

        // Encrypt locally before sending
        let encryptedData = try encryptionService.encrypt(thoughtData, using: key)

        // Create CKRecord with encrypted payload
        let record = CKRecord(recordType: "EncryptedThought")
        record["id"] = thought.id.uuidString
        record["encryptedPayload"] = encryptedData as CKRecordValue
        record["createdAt"] = thought.createdAt  // Unencrypted for sorting

        // Include content hash for deduplication (not revealing content)
        let contentHash = SHA256.hash(data: thoughtData)
        record["contentHash"] = Data(contentHash).base64EncodedString()

        try await privateDatabase.save(record)
    }

    /// Fetch and decrypt thoughts
    func fetchThoughts(since date: Date? = nil) async throws -> [ThoughtNode] {
        guard let key = syncKey else {
            throw CloudSyncError.notConfigured
        }

        var predicate = NSPredicate(value: true)
        if let date = date {
            predicate = NSPredicate(format: "createdAt > %@", date as NSDate)
        }

        let query = CKQuery(recordType: "EncryptedThought", predicate: predicate)
        query.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]

        let (results, _) = try await privateDatabase.records(matching: query)

        var thoughts: [ThoughtNode] = []

        for (_, result) in results {
            if case .success(let record) = result,
               let encryptedData = record["encryptedPayload"] as? Data {
                // Decrypt locally
                let decrypted = try encryptionService.decrypt(encryptedData, using: key)
                let decoder = JSONDecoder()
                let thought = try decoder.decode(ThoughtNode.self, from: decrypted)
                thoughts.append(thought)
            }
        }

        return thoughts
    }

    // MARK: - Sync State Management

    /// Set up subscription for real-time sync
    func setupSyncSubscription() async throws {
        let subscription = CKQuerySubscription(
            recordType: "EncryptedThought",
            predicate: NSPredicate(value: true),
            options: [.firesOnRecordCreation, .firesOnRecordUpdate]
        )

        let notification = CKSubscription.NotificationInfo()
        notification.shouldSendContentAvailable = true  // Silent push
        subscription.notificationInfo = notification

        try await privateDatabase.save(subscription)
    }

    // MARK: - Key Sync Across Devices

    /// Store encrypted sync key in CloudKit for device pairing
    /// The key is encrypted with device-specific key derived from password
    func storeSyncKeyBundle(_ bundle: Data) async throws {
        let record = CKRecord(recordType: "KeyBundle")
        record["bundle"] = bundle as CKRecordValue
        record["createdAt"] = Date()

        try await privateDatabase.save(record)
    }

    /// Retrieve sync key bundle for new device setup
    func retrieveSyncKeyBundle() async throws -> Data? {
        let query = CKQuery(
            recordType: "KeyBundle",
            predicate: NSPredicate(value: true)
        )
        query.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]

        let (results, _) = try await privateDatabase.records(
            matching: query,
            resultsLimit: 1
        )

        guard let (_, result) = results.first,
              case .success(let record) = result else {
            return nil
        }

        return record["bundle"] as? Data
    }
}

enum CloudSyncError: Error {
    case notConfigured
    case encryptionFailed
    case decryptionFailed
    case recordNotFound
}
```

### 4.2 Zero-Knowledge Architecture

In a zero-knowledge system, the server cannot read user data:

```swift
// MARK: - Zero-Knowledge Sync Protocol

protocol ZeroKnowledgeSyncProtocol {
    /// Server never sees plaintext
    func encryptBeforeUpload(_ data: Data) throws -> Data

    /// Client decrypts after download
    func decryptAfterDownload(_ encrypted: Data) throws -> Data

    /// Key never leaves device
    var keyNeverTransmitted: Bool { get }
}

final class ZeroKnowledgeSync: ZeroKnowledgeSyncProtocol {

    private let localKey: SymmetricKey

    var keyNeverTransmitted: Bool { true }

    init(masterPassword: String) throws {
        // Key derived from password - never sent to server
        let salt = Self.getOrCreateDeviceSalt()
        let encryptionService = EncryptionService()
        self.localKey = try encryptionService.deriveKeyFromPassword(
            masterPassword,
            salt: salt
        )
    }

    func encryptBeforeUpload(_ data: Data) throws -> Data {
        let service = EncryptionService()
        return try service.encrypt(data, using: localKey)
    }

    func decryptAfterDownload(_ encrypted: Data) throws -> Data {
        let service = EncryptionService()
        return try service.decrypt(encrypted, using: localKey)
    }

    private static func getOrCreateDeviceSalt() -> Data {
        let keychain = KeychainService()

        if let existing = try? keychain.retrieve(key: "device_salt") {
            return existing
        }

        // Generate new salt for this device
        var salt = Data(count: 32)
        _ = salt.withUnsafeMutableBytes { ptr in
            SecRandomCopyBytes(kSecRandomDefault, 32, ptr.baseAddress!)
        }

        try? keychain.store(salt, key: "device_salt")
        return salt
    }
}
```

### 4.3 Multi-Device Key Management

```swift
// MARK: - Multi-Device Key Coordinator

final class MultiDeviceKeyCoordinator {

    private let keychain: KeychainService
    private let cloudKit: SecureCloudKitManager
    private let encryption: EncryptionService

    /// Onboard new device by transferring sync key
    func onboardNewDevice(
        existingDeviceApproval: () async throws -> Bool
    ) async throws {
        // Step 1: Generate ephemeral key pair for this device
        let deviceKeyPair = P256.KeyAgreement.PrivateKey()
        let devicePublicKey = deviceKeyPair.publicKey

        // Step 2: Display QR code / share code with existing device
        let pairingCode = generatePairingCode(publicKey: devicePublicKey)

        // Step 3: Wait for approval from existing device
        guard try await existingDeviceApproval() else {
            throw KeyTransferError.denied
        }

        // Step 4: Receive encrypted key bundle from existing device
        guard let encryptedBundle = try await cloudKit.retrieveSyncKeyBundle() else {
            throw KeyTransferError.bundleNotFound
        }

        // Step 5: Decrypt bundle using ECDH
        let syncKey = try decryptKeyBundle(
            encryptedBundle,
            using: deviceKeyPair
        )

        // Step 6: Store sync key in local keychain
        try keychain.store(
            syncKey.withUnsafeBytes { Data($0) },
            key: "sync_key",
            accessibility: .afterFirstUnlockThisDeviceOnly,
            requireBiometric: true
        )
    }

    /// Approve new device from existing device
    func approveNewDevice(newDevicePublicKey: P256.KeyAgreement.PublicKey) async throws {
        // Retrieve existing sync key
        let syncKeyData = try keychain.retrieve(key: "sync_key")
        let syncKey = SymmetricKey(data: syncKeyData)

        // Encrypt sync key for new device
        let encryptedBundle = try createKeyBundle(
            syncKey: syncKey,
            for: newDevicePublicKey
        )

        // Upload encrypted bundle
        try await cloudKit.storeSyncKeyBundle(encryptedBundle)
    }

    private func generatePairingCode(publicKey: P256.KeyAgreement.PublicKey) -> String {
        // Encode public key as base64 for display
        publicKey.rawRepresentation.base64EncodedString()
    }

    private func createKeyBundle(
        syncKey: SymmetricKey,
        for recipientPublicKey: P256.KeyAgreement.PublicKey
    ) throws -> Data {
        // Create ephemeral key for ECDH
        let ephemeralKey = P256.KeyAgreement.PrivateKey()

        // Derive shared secret
        let sharedSecret = try ephemeralKey.sharedSecretFromKeyAgreement(
            with: recipientPublicKey
        )

        // Derive transport key
        let transportKey = sharedSecret.hkdfDerivedSymmetricKey(
            using: SHA256.self,
            salt: Data(),
            sharedInfo: "MYND-DEVICE-PAIRING".data(using: .utf8)!,
            outputByteCount: 32
        )

        // Encrypt sync key
        let syncKeyData = syncKey.withUnsafeBytes { Data($0) }
        let encryptedKey = try encryption.encrypt(syncKeyData, using: transportKey)

        // Bundle: ephemeral public key + encrypted sync key
        var bundle = Data()
        bundle.append(ephemeralKey.publicKey.rawRepresentation)
        bundle.append(encryptedKey)

        return bundle
    }

    private func decryptKeyBundle(
        _ bundle: Data,
        using deviceKey: P256.KeyAgreement.PrivateKey
    ) throws -> SymmetricKey {
        // Extract ephemeral public key (first 65 bytes for uncompressed P256)
        let ephemeralPublicKeyData = bundle.prefix(65)
        let ephemeralPublicKey = try P256.KeyAgreement.PublicKey(
            rawRepresentation: ephemeralPublicKeyData
        )

        // Derive shared secret
        let sharedSecret = try deviceKey.sharedSecretFromKeyAgreement(
            with: ephemeralPublicKey
        )

        // Derive transport key
        let transportKey = sharedSecret.hkdfDerivedSymmetricKey(
            using: SHA256.self,
            salt: Data(),
            sharedInfo: "MYND-DEVICE-PAIRING".data(using: .utf8)!,
            outputByteCount: 32
        )

        // Decrypt sync key
        let encryptedKey = bundle.dropFirst(65)
        let syncKeyData = try encryption.decrypt(Data(encryptedKey), using: transportKey)

        return SymmetricKey(data: syncKeyData)
    }
}

enum KeyTransferError: Error {
    case denied
    case bundleNotFound
    case decryptionFailed
}
```

---

## 5. API Security

### 5.1 API Key Protection

**CRITICAL**: Never embed API keys in the app bundle. They can be extracted trivially.

```swift
// MARK: - API Key Manager

final class APIKeyManager {

    private let keychain: KeychainService
    private static let anthropicKeyIdentifier = "com.mynd.api.anthropic"
    private static let openaiKeyIdentifier = "com.mynd.api.openai"

    /// Store API key securely
    func storeAPIKey(_ key: String, for service: APIService) throws {
        guard key.count > 10 else {  // Basic validation
            throw APIKeyError.invalidFormat
        }

        try keychain.store(
            Data(key.utf8),
            key: service.identifier,
            accessibility: .whenUnlockedThisDeviceOnly,
            requireBiometric: false  // API keys used in background
        )
    }

    /// Retrieve API key
    func retrieveAPIKey(for service: APIService) throws -> String {
        let data = try keychain.retrieve(key: service.identifier)
        guard let key = String(data: data, encoding: .utf8) else {
            throw APIKeyError.corruptedData
        }
        return key
    }

    /// Check if API key exists
    func hasAPIKey(for service: APIService) -> Bool {
        (try? keychain.retrieve(key: service.identifier)) != nil
    }

    /// Delete API key
    func deleteAPIKey(for service: APIService) throws {
        try keychain.delete(key: service.identifier)
    }
}

enum APIService {
    case anthropic
    case openai
    case elevenLabs

    var identifier: String {
        switch self {
        case .anthropic: return "com.mynd.api.anthropic"
        case .openai: return "com.mynd.api.openai"
        case .elevenLabs: return "com.mynd.api.elevenlabs"
        }
    }
}

enum APIKeyError: Error {
    case invalidFormat
    case corruptedData
    case notFound
}
```

### 5.2 Certificate Pinning

Prevent man-in-the-middle attacks by pinning to expected certificates:

```swift
import Foundation
import CryptoKit

// MARK: - Certificate Pinning

final class CertificatePinningDelegate: NSObject, URLSessionDelegate {

    private let pinnedHashes: [String: Set<String>]

    /// Initialize with domain-to-hash mappings
    /// Hashes are SHA256 of SubjectPublicKeyInfo
    init(pins: [String: Set<String>]) {
        self.pinnedHashes = pins
        super.init()
    }

    /// Convenience initializer with known Anthropic/OpenAI pins
    static func withDefaultPins() -> CertificatePinningDelegate {
        CertificatePinningDelegate(pins: [
            "api.anthropic.com": [
                // Primary pin (current certificate)
                "abc123...",  // Replace with actual SPKI hash
                // Backup pin (next certificate)
                "def456..."   // Replace with actual SPKI hash
            ],
            "api.openai.com": [
                "ghi789...",
                "jkl012..."
            ]
        ])
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let serverTrust = challenge.protectionSpace.serverTrust,
              let pinnedSet = pinnedHashes[challenge.protectionSpace.host] else {
            // No pinning configured for this host
            completionHandler(.performDefaultHandling, nil)
            return
        }

        // Validate certificate chain
        var error: CFError?
        let isValid = SecTrustEvaluateWithError(serverTrust, &error)

        guard isValid else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        // Check if any certificate in chain matches our pins
        let certificateCount = SecTrustGetCertificateCount(serverTrust)

        for index in 0..<certificateCount {
            guard let certificate = SecTrustGetCertificateAtIndex(serverTrust, index) else {
                continue
            }

            let publicKeyHash = hashPublicKey(of: certificate)

            if pinnedSet.contains(publicKeyHash) {
                completionHandler(.useCredential, URLCredential(trust: serverTrust))
                return
            }
        }

        // No matching pin found - reject connection
        completionHandler(.cancelAuthenticationChallenge, nil)
    }

    private func hashPublicKey(of certificate: SecCertificate) -> String {
        guard let publicKey = SecCertificateCopyKey(certificate),
              let publicKeyData = SecKeyCopyExternalRepresentation(publicKey, nil) as Data? else {
            return ""
        }

        let hash = SHA256.hash(data: publicKeyData)
        return Data(hash).base64EncodedString()
    }
}

// MARK: - Pinned URL Session

extension URLSession {

    static func pinnedSession() -> URLSession {
        let delegate = CertificatePinningDelegate.withDefaultPins()
        let configuration = URLSessionConfiguration.default
        configuration.tlsMinimumSupportedProtocolVersion = .TLSv13

        return URLSession(
            configuration: configuration,
            delegate: delegate,
            delegateQueue: nil
        )
    }
}
```

### 5.3 Token Refresh Pattern

```swift
// MARK: - Token Manager

actor TokenManager {

    private var accessToken: String?
    private var refreshToken: String?
    private var expiresAt: Date?

    private let keychain: KeychainService
    private let apiClient: APIClient

    /// Get valid access token, refreshing if necessary
    func getValidToken() async throws -> String {
        // Check if current token is still valid
        if let token = accessToken,
           let expiry = expiresAt,
           expiry > Date().addingTimeInterval(60) {  // 60 second buffer
            return token
        }

        // Try to refresh
        if let refresh = refreshToken {
            return try await refreshAccessToken(using: refresh)
        }

        throw TokenError.noValidToken
    }

    /// Store tokens from login/auth response
    func storeTokens(access: String, refresh: String, expiresIn: TimeInterval) throws {
        self.accessToken = access
        self.refreshToken = refresh
        self.expiresAt = Date().addingTimeInterval(expiresIn)

        // Persist refresh token securely
        try keychain.store(
            Data(refresh.utf8),
            key: "refresh_token",
            accessibility: .afterFirstUnlockThisDeviceOnly
        )
    }

    /// Clear all tokens (logout)
    func clearTokens() throws {
        accessToken = nil
        refreshToken = nil
        expiresAt = nil
        try keychain.delete(key: "refresh_token")
    }

    private func refreshAccessToken(using refresh: String) async throws -> String {
        let response = try await apiClient.refresh(token: refresh)

        self.accessToken = response.accessToken
        self.expiresAt = Date().addingTimeInterval(response.expiresIn)

        if let newRefresh = response.refreshToken {
            self.refreshToken = newRefresh
            try keychain.store(
                Data(newRefresh.utf8),
                key: "refresh_token",
                accessibility: .afterFirstUnlockThisDeviceOnly
            )
        }

        return response.accessToken
    }
}

enum TokenError: Error {
    case noValidToken
    case refreshFailed
}
```

### 5.4 Network Security Configuration

Configure App Transport Security in Info.plist:

```xml
<!-- Info.plist -->
<key>NSAppTransportSecurity</key>
<dict>
    <!-- Require HTTPS for all connections -->
    <key>NSAllowsArbitraryLoads</key>
    <false/>

    <!-- Specific domain configurations if needed -->
    <key>NSExceptionDomains</key>
    <dict>
        <key>api.anthropic.com</key>
        <dict>
            <key>NSExceptionMinimumTLSVersion</key>
            <string>TLSv1.3</string>
            <key>NSExceptionRequiresForwardSecrecy</key>
            <true/>
        </dict>
    </dict>
</dict>
```

---

## 6. GDPR & Privacy Compliance

### 6.1 Data Minimization Implementation

```swift
// MARK: - Privacy-Conscious Data Model

/// Only collect data that is strictly necessary
struct MinimalThought: Codable {
    let id: UUID
    let content: String  // User's actual thought - necessary
    let createdAt: Date  // Timestamp - necessary for ordering

    // NOT collected:
    // - Precise location (only timezone if needed)
    // - Device identifiers
    // - IP addresses
    // - Usage analytics beyond essentials
}

// MARK: - Data Minimization Service

final class DataMinimizationService {

    /// Strip unnecessary metadata before storage
    func minimize(_ rawThought: RawThoughtCapture) -> MinimalThought {
        MinimalThought(
            id: rawThought.id,
            content: rawThought.transcription,
            createdAt: rawThought.timestamp
        )
        // Discarded: rawThought.audioData, rawThought.deviceInfo, etc.
    }

    /// Configure analytics to respect privacy
    func configureAnalytics() {
        // Only track essential metrics
        Analytics.shared.configure(
            collectDeviceId: false,
            collectPreciseLocation: false,
            collectIPAddress: false,
            dataRetentionDays: 30
        )
    }
}
```

### 6.2 Right to Deletion Implementation

```swift
// MARK: - Data Deletion Service

final class DataDeletionService {

    private let swiftDataStore: ModelContext
    private let cloudKit: SecureCloudKitManager
    private let keychain: KeychainService

    /// Complete account deletion (GDPR Article 17)
    func deleteAllUserData() async throws {
        // Step 1: Delete from local SwiftData
        try await deleteLocalData()

        // Step 2: Delete from CloudKit
        try await deleteCloudData()

        // Step 3: Delete cached files
        try deleteCachedFiles()

        // Step 4: Clear keychain (except device identity)
        try clearSensitiveKeychain()

        // Step 5: Request deletion from third-party services
        try await requestThirdPartyDeletion()

        // Step 6: Log deletion for compliance audit
        logDeletionRequest()
    }

    private func deleteLocalData() async throws {
        // Delete all user content
        try swiftDataStore.delete(model: ThoughtNode.self)
        try swiftDataStore.delete(model: Edge.self)
        try swiftDataStore.delete(model: Message.self)
        try swiftDataStore.delete(model: ConversationSession.self)
        try swiftDataStore.delete(model: MemoryItem.self)
        try swiftDataStore.delete(model: UserPattern.self)

        try swiftDataStore.save()
    }

    private func deleteCloudData() async throws {
        // Query all user records
        let query = CKQuery(
            recordType: "EncryptedThought",
            predicate: NSPredicate(value: true)
        )

        let (results, _) = try await cloudKit.privateDatabase.records(matching: query)

        // Delete each record
        for (recordID, _) in results {
            try await cloudKit.privateDatabase.deleteRecord(withID: recordID)
        }

        // Delete key bundles
        let keyQuery = CKQuery(
            recordType: "KeyBundle",
            predicate: NSPredicate(value: true)
        )
        let (keyResults, _) = try await cloudKit.privateDatabase.records(matching: keyQuery)

        for (recordID, _) in keyResults {
            try await cloudKit.privateDatabase.deleteRecord(withID: recordID)
        }
    }

    private func deleteCachedFiles() throws {
        let cacheURL = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        ).first!

        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheURL,
            includingPropertiesForKeys: nil
        )

        for url in contents {
            try FileManager.default.removeItem(at: url)
        }
    }

    private func clearSensitiveKeychain() throws {
        try keychain.delete(key: "sync_key")
        try keychain.delete(key: "e2ee_salt")
        try keychain.delete(key: "e2ee_verify")
        try keychain.delete(key: "com.mynd.api.anthropic")
        try keychain.delete(key: "com.mynd.api.openai")
        try keychain.delete(key: "refresh_token")
    }

    private func requestThirdPartyDeletion() async throws {
        // If using managed API service, request data deletion
        // Claude/OpenAI typically don't retain conversation data
        // but document the request for compliance
    }

    private func logDeletionRequest() {
        // Log deletion for audit trail (store locally only)
        let entry = AuditLogEntry(
            action: .accountDeletion,
            timestamp: Date(),
            details: "User requested complete data deletion"
        )
        AuditLogger.shared.log(entry)
    }
}
```

### 6.3 Data Export (Portability)

```swift
// MARK: - Data Export Service

final class DataExportService {

    private let modelContext: ModelContext
    private let encryptionService: EncryptionService

    /// Export all user data in portable format (GDPR Article 20)
    func exportAllData() async throws -> DataExportBundle {

        // Fetch all user data
        let thoughts = try await fetchAllThoughts()
        let sessions = try await fetchAllSessions()
        let patterns = try await fetchAllPatterns()

        // Create export bundle
        let bundle = DataExportBundle(
            exportDate: Date(),
            format: "JSON",
            version: "1.0",
            data: ExportData(
                thoughts: thoughts,
                conversations: sessions,
                learnedPatterns: patterns
            )
        )

        return bundle
    }

    /// Export to JSON file
    func exportToFile() async throws -> URL {
        let bundle = try await exportAllData()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(bundle)

        // Save to documents
        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!

        let exportURL = documentsURL.appendingPathComponent(
            "MYND_Export_\(ISO8601DateFormatter().string(from: Date())).json"
        )

        try data.write(to: exportURL)

        return exportURL
    }

    /// Export encrypted (for secure transfer)
    func exportEncrypted(password: String) async throws -> URL {
        let bundle = try await exportAllData()

        let encoder = JSONEncoder()
        let data = try encoder.encode(bundle)

        // Derive key from password
        let salt = encryptionService.generateSalt()
        let key = try encryptionService.deriveKeyFromPassword(password, salt: salt)

        // Encrypt
        let encrypted = try encryptionService.encrypt(data, using: key)

        // Bundle: salt + encrypted data
        var exportData = Data()
        exportData.append(salt)
        exportData.append(encrypted)

        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!

        let exportURL = documentsURL.appendingPathComponent(
            "MYND_Export_\(ISO8601DateFormatter().string(from: Date())).encrypted"
        )

        try exportData.write(to: exportURL)

        return exportURL
    }
}

struct DataExportBundle: Codable {
    let exportDate: Date
    let format: String
    let version: String
    let data: ExportData
}

struct ExportData: Codable {
    let thoughts: [ThoughtExport]
    let conversations: [ConversationExport]
    let learnedPatterns: [PatternExport]
}
```

### 6.4 Consent Management

```swift
// MARK: - Consent Manager

final class ConsentManager: ObservableObject {

    @Published private(set) var consents: [ConsentType: ConsentRecord] = [:]

    private let storage: UserDefaults

    init(storage: UserDefaults = .standard) {
        self.storage = storage
        loadConsents()
    }

    /// Request consent for specific purpose
    func requestConsent(
        for type: ConsentType,
        presenter: ConsentPresenter
    ) async -> Bool {
        let granted = await presenter.showConsentDialog(for: type)

        if granted {
            recordConsent(type, granted: true)
        }

        return granted
    }

    /// Check if consent was granted
    func hasConsent(for type: ConsentType) -> Bool {
        consents[type]?.granted ?? false
    }

    /// Withdraw consent (user can revoke at any time)
    func withdrawConsent(for type: ConsentType) {
        recordConsent(type, granted: false)

        // Trigger any necessary data deletion
        Task {
            await handleConsentWithdrawal(type)
        }
    }

    /// Get all consent records for compliance reporting
    func getConsentAuditLog() -> [ConsentRecord] {
        Array(consents.values)
    }

    private func recordConsent(_ type: ConsentType, granted: Bool) {
        let record = ConsentRecord(
            type: type,
            granted: granted,
            timestamp: Date(),
            version: ConsentManager.currentPolicyVersion
        )

        consents[type] = record
        saveConsents()
    }

    private func handleConsentWithdrawal(_ type: ConsentType) async {
        switch type {
        case .analytics:
            // Stop analytics collection
            Analytics.shared.disable()

        case .cloudSync:
            // Delete cloud data
            try? await DataDeletionService().deleteCloudData()

        case .aiProcessing:
            // Switch to local-only mode
            UserDefaults.standard.set(true, forKey: "localOnlyMode")
        }
    }

    private static let currentPolicyVersion = "1.0"

    private func loadConsents() {
        if let data = storage.data(forKey: "consents"),
           let decoded = try? JSONDecoder().decode([ConsentType: ConsentRecord].self, from: data) {
            consents = decoded
        }
    }

    private func saveConsents() {
        if let data = try? JSONEncoder().encode(consents) {
            storage.set(data, forKey: "consents")
        }
    }
}

enum ConsentType: String, Codable, CaseIterable {
    case essentialFunctionality = "Essential app functionality"
    case analytics = "Anonymous usage analytics"
    case cloudSync = "Cloud sync across devices"
    case aiProcessing = "AI processing of thoughts"
    case voiceRecording = "Voice recording for transcription"
}

struct ConsentRecord: Codable {
    let type: ConsentType
    let granted: Bool
    let timestamp: Date
    let version: String
}
```

### 6.5 Privacy Policy Requirements

Key elements that must be in MYND's privacy policy:

1. **Data Collection**: Specify exactly what data is collected
2. **Purpose**: Why each data type is collected
3. **Storage**: Where data is stored (device, CloudKit)
4. **Third Parties**: List of third parties (Anthropic, OpenAI if used)
5. **Retention**: How long data is kept
6. **User Rights**: Right to access, delete, export
7. **Contact**: How to submit privacy requests
8. **Updates**: How policy changes are communicated

### 6.6 Apple App Store Privacy Labels

Configure these in App Store Connect:

```
Data Linked to You:
- Contact Info: Name, Email (if account created)
- User Content: Voice recordings (processed, not stored), Thoughts (stored)
- Identifiers: User ID (if account created)

Data Not Linked to You:
- Usage Data: Product interaction (anonymous analytics)
- Diagnostics: Crash logs

Data Not Collected:
- Location
- Financial Info
- Health & Fitness
- Browsing History
- Search History
- Contacts
- Photos/Videos
```

---

## 7. AI/LLM Privacy Considerations

### 7.1 Data Sent to AI Providers

**Claude API (Anthropic)**:
- Sends: User prompt + system prompt + conversation history
- Does NOT send: File contents, device info, user identity
- Retention: Not used for training (API terms)

**OpenAI API**:
- Similar to Claude
- Opt-out of training available via API settings

```swift
// MARK: - AI Privacy Manager

final class AIPrivacyManager {

    /// Sanitize data before sending to AI provider
    func prepareForAI(_ text: String, settings: PrivacySettings) -> String {
        var sanitized = text

        if settings.stripPII {
            sanitized = stripPII(sanitized)
        }

        if settings.generalizeNames {
            sanitized = generalizeNames(sanitized)
        }

        return sanitized
    }

    private func stripPII(_ text: String) -> String {
        var result = text

        // Remove email addresses
        result = result.replacingOccurrences(
            of: #"[\w.]+@[\w.]+"#,
            with: "[email]",
            options: .regularExpression
        )

        // Remove phone numbers
        result = result.replacingOccurrences(
            of: #"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"#,
            with: "[phone]",
            options: .regularExpression
        )

        // Remove SSN patterns
        result = result.replacingOccurrences(
            of: #"\b\d{3}[-]?\d{2}[-]?\d{4}\b"#,
            with: "[ssn]",
            options: .regularExpression
        )

        // Remove credit card numbers
        result = result.replacingOccurrences(
            of: #"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"#,
            with: "[card]",
            options: .regularExpression
        )

        return result
    }

    private func generalizeNames(_ text: String) -> String {
        // Use NLTagger to identify names
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text

        var result = text
        var replacements: [(Range<String.Index>, String)] = []

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType) { tag, range in
            if tag == .personalName {
                replacements.append((range, "[person]"))
            }
            return true
        }

        // Apply replacements in reverse order to maintain indices
        for (range, replacement) in replacements.reversed() {
            result.replaceSubrange(range, with: replacement)
        }

        return result
    }
}
```

### 7.2 OpenAI/Anthropic Data Policies

**Anthropic Claude API** (as of 2025):
- Does NOT use API data for training by default
- 30-day retention for abuse monitoring
- Can request immediate deletion
- HIPAA BAA available for healthcare

**OpenAI API** (as of 2025):
- Does NOT use API data for training by default
- Opt-out available in organization settings
- Data retention configurable
- SOC 2 Type II certified

```swift
// MARK: - API Headers for Privacy

extension ClaudeClient {

    func buildPrivacyHeaders() -> [String: String] {
        [
            "x-api-key": apiKey,
            "anthropic-version": "2024-01-01",
            // Request no training data usage
            "anthropic-beta": "training-opt-out"
        ]
    }
}

extension OpenAIClient {

    func configurePrivacy() -> URLRequest {
        var request = baseRequest()
        // OpenAI respects org-level training data settings
        // Configure in dashboard, not per-request
        return request
    }
}
```

### 7.3 Opt-Out of Training Data

```swift
// MARK: - Training Data Opt-Out

final class TrainingDataOptOutManager {

    /// Display clear opt-out information to user
    func getOptOutStatus() -> [AIProvider: OptOutStatus] {
        [
            .anthropic: .automaticOptOut,  // Claude API doesn't train on data
            .openai: .requiresOrgSetting   // Set in OpenAI dashboard
        ]
    }

    /// Show user what data is sent
    func generateDataTransparencyReport(for session: ConversationSession) -> TransparencyReport {
        TransparencyReport(
            provider: .anthropic,
            dataSent: [
                "System prompt (Axel personality)",
                "Your messages in this conversation",
                "Relevant memories (up to 5)"
            ],
            dataNotSent: [
                "Your name or email",
                "Device information",
                "Location data",
                "Other conversations",
                "Voice recordings (only transcripts sent)"
            ],
            retention: "30 days for abuse monitoring, then deleted",
            usedForTraining: false
        )
    }
}

enum AIProvider {
    case anthropic
    case openai
    case local
}

enum OptOutStatus {
    case automaticOptOut
    case requiresOrgSetting
    case notApplicable
}

struct TransparencyReport {
    let provider: AIProvider
    let dataSent: [String]
    let dataNotSent: [String]
    let retention: String
    let usedForTraining: Bool
}
```

---

## 8. Audit & Transparency

### 8.1 Activity Logging

```swift
// MARK: - Audit Logger

final class AuditLogger {

    static let shared = AuditLogger()

    private let logFile: URL
    private let encryptionService: EncryptionService
    private var logKey: SymmetricKey?

    private init() {
        let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        logFile = documentsPath.appendingPathComponent("audit.log.encrypted")
        encryptionService = EncryptionService()
    }

    /// Log an auditable action
    func log(_ entry: AuditLogEntry) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        guard let entryData = try? encoder.encode(entry) else { return }

        // Append to log file
        appendToLog(entryData)
    }

    /// Get audit log for user review
    func getRecentEntries(limit: Int = 100) -> [AuditLogEntry] {
        guard let entries = readLogEntries() else { return [] }
        return Array(entries.suffix(limit))
    }

    /// Export audit log for compliance
    func exportAuditLog() throws -> Data {
        guard let entries = readLogEntries() else {
            return Data()
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601

        return try encoder.encode(entries)
    }

    private func appendToLog(_ data: Data) {
        // In production: encrypt before writing
        // Simplified for example
        if let handle = try? FileHandle(forWritingTo: logFile) {
            handle.seekToEndOfFile()
            handle.write(data)
            handle.write("\n".data(using: .utf8)!)
            handle.closeFile()
        } else {
            try? data.write(to: logFile)
        }
    }

    private func readLogEntries() -> [AuditLogEntry]? {
        guard let data = try? Data(contentsOf: logFile) else { return nil }

        let lines = String(data: data, encoding: .utf8)?
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return lines?.compactMap { line in
            try? decoder.decode(AuditLogEntry.self, from: Data(line.utf8))
        }
    }
}

struct AuditLogEntry: Codable {
    let id: UUID
    let action: AuditAction
    let timestamp: Date
    let details: String
    let dataAccessed: [String]?

    init(
        action: AuditAction,
        timestamp: Date = Date(),
        details: String,
        dataAccessed: [String]? = nil
    ) {
        self.id = UUID()
        self.action = action
        self.timestamp = timestamp
        self.details = details
        self.dataAccessed = dataAccessed
    }
}

enum AuditAction: String, Codable {
    case appLaunched
    case thoughtCreated
    case thoughtDeleted
    case dataExported
    case dataImported
    case accountDeletion
    case cloudSyncEnabled
    case cloudSyncDisabled
    case apiKeyAdded
    case apiKeyRemoved
    case consentGranted
    case consentWithdrawn
    case dataAccessRequest
}
```

### 8.2 User Data Access Requests

```swift
// MARK: - Data Access Request Handler

final class DataAccessRequestHandler {

    private let modelContext: ModelContext
    private let auditLogger: AuditLogger

    /// Handle user's data access request (GDPR Article 15)
    func handleAccessRequest() async throws -> DataAccessReport {
        // Log the request
        auditLogger.log(AuditLogEntry(
            action: .dataAccessRequest,
            details: "User requested access to their data"
        ))

        // Compile all data
        let report = DataAccessReport(
            requestDate: Date(),
            personalData: try await compilePersonalData(),
            processingPurposes: getProcessingPurposes(),
            thirdPartyRecipients: getThirdPartyRecipients(),
            retentionPeriods: getRetentionPeriods(),
            userRights: getUserRights()
        )

        return report
    }

    private func compilePersonalData() async throws -> PersonalDataSummary {
        // Count all data categories
        let thoughtCount = try modelContext.fetchCount(FetchDescriptor<ThoughtNode>())
        let sessionCount = try modelContext.fetchCount(FetchDescriptor<ConversationSession>())
        let messageCount = try modelContext.fetchCount(FetchDescriptor<Message>())

        return PersonalDataSummary(
            thoughtsCount: thoughtCount,
            conversationsCount: sessionCount,
            messagesCount: messageCount,
            oldestDataDate: try await getOldestDataDate(),
            newestDataDate: Date()
        )
    }

    private func getProcessingPurposes() -> [ProcessingPurpose] {
        [
            ProcessingPurpose(
                category: "Thought storage",
                purpose: "Store and organize your captured thoughts",
                legalBasis: "Consent / Contract performance"
            ),
            ProcessingPurpose(
                category: "AI conversation",
                purpose: "Provide intelligent responses to your inputs",
                legalBasis: "Consent"
            ),
            ProcessingPurpose(
                category: "Pattern learning",
                purpose: "Learn your preferences to provide better assistance",
                legalBasis: "Consent"
            )
        ]
    }

    private func getThirdPartyRecipients() -> [ThirdPartyRecipient] {
        [
            ThirdPartyRecipient(
                name: "Anthropic",
                purpose: "AI conversation processing",
                dataShared: "Conversation content (not stored)",
                location: "United States",
                safeguards: "Standard Contractual Clauses"
            ),
            ThirdPartyRecipient(
                name: "Apple (CloudKit)",
                purpose: "Cloud storage and sync",
                dataShared: "Encrypted thought data",
                location: "United States / EU",
                safeguards: "Apple Privacy Policy, E2E encryption"
            )
        ]
    }

    private func getRetentionPeriods() -> [RetentionPeriod] {
        [
            RetentionPeriod(
                dataType: "Thoughts and conversations",
                period: "Until you delete them",
                deletionMethod: "User-initiated or account deletion"
            ),
            RetentionPeriod(
                dataType: "Analytics data",
                period: "30 days",
                deletionMethod: "Automatic expiration"
            ),
            RetentionPeriod(
                dataType: "Crash logs",
                period: "90 days",
                deletionMethod: "Automatic expiration"
            )
        ]
    }

    private func getUserRights() -> [UserRight] {
        [
            UserRight(
                right: "Access",
                description: "View all data we hold about you",
                howToExercise: "Settings > Privacy > Export Data"
            ),
            UserRight(
                right: "Rectification",
                description: "Correct inaccurate data",
                howToExercise: "Edit thoughts directly in the app"
            ),
            UserRight(
                right: "Erasure",
                description: "Delete all your data",
                howToExercise: "Settings > Account > Delete Account"
            ),
            UserRight(
                right: "Portability",
                description: "Export your data in portable format",
                howToExercise: "Settings > Privacy > Export Data"
            ),
            UserRight(
                right: "Objection",
                description: "Object to specific processing",
                howToExercise: "Contact support@mynd.app"
            )
        ]
    }
}

struct DataAccessReport: Codable {
    let requestDate: Date
    let personalData: PersonalDataSummary
    let processingPurposes: [ProcessingPurpose]
    let thirdPartyRecipients: [ThirdPartyRecipient]
    let retentionPeriods: [RetentionPeriod]
    let userRights: [UserRight]
}
```

### 8.3 Third-Party Security Audits

Recommendations for MYND:

1. **Pre-Launch**: Self-audit using OWASP Mobile Security Testing Guide
2. **Post-Launch (3 months)**: Penetration testing by certified firm
3. **Annual**: SOC 2 Type I audit if handling sensitive data
4. **Continuous**: Automated security scanning in CI/CD

---

## 9. App Transport Security

### 9.1 HTTPS Enforcement

```swift
// MARK: - Network Security Configuration

final class NetworkSecurityConfiguration {

    /// Create secure URLSession configuration
    static func secureConfiguration() -> URLSessionConfiguration {
        let config = URLSessionConfiguration.default

        // Enforce TLS 1.3 minimum
        config.tlsMinimumSupportedProtocolVersion = .TLSv13

        // Disable insecure protocols
        config.httpShouldSetCookies = false

        // Set reasonable timeouts
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60

        return config
    }

    /// Validate URL is HTTPS before making request
    static func validateSecureURL(_ url: URL) throws {
        guard url.scheme == "https" else {
            throw NetworkSecurityError.insecureConnection
        }
    }
}

enum NetworkSecurityError: Error {
    case insecureConnection
    case invalidCertificate
    case tlsVersionTooLow
}
```

### 9.2 TLS 1.3 Requirements

Enforce TLS 1.3 for all connections:

```swift
// MARK: - TLS Configuration

extension URLSessionConfiguration {

    /// Configure for maximum security
    static var maximumSecurity: URLSessionConfiguration {
        let config = URLSessionConfiguration.default

        // Require TLS 1.3
        config.tlsMinimumSupportedProtocolVersion = .TLSv13
        config.tlsMaximumSupportedProtocolVersion = .TLSv13

        return config
    }
}

// Info.plist configuration
/*
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <false/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>api.anthropic.com</key>
        <dict>
            <key>NSExceptionMinimumTLSVersion</key>
            <string>TLSv1.3</string>
        </dict>
        <key>api.openai.com</key>
        <dict>
            <key>NSExceptionMinimumTLSVersion</key>
            <string>TLSv1.3</string>
        </dict>
    </dict>
</dict>
*/
```

### 9.3 Network Debugging Protection

Prevent traffic interception in debug builds:

```swift
// MARK: - Debug Protection

#if DEBUG
final class DebugProtection {

    /// Check if device is compromised
    static var isCompromised: Bool {
        // Check for common jailbreak indicators
        let jailbreakPaths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/"
        ]

        for path in jailbreakPaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }

        // Check if app can write outside sandbox
        let testPath = "/private/jailbreak_test.txt"
        do {
            try "test".write(toFile: testPath, atomically: true, encoding: .utf8)
            try FileManager.default.removeItem(atPath: testPath)
            return true  // Shouldn't be able to write here
        } catch {
            // Expected behavior
        }

        return false
    }

    /// Check for proxy/MITM
    static var isProxyDetected: Bool {
        guard let proxySettings = CFNetworkCopySystemProxySettings()?.takeRetainedValue() as? [String: Any] else {
            return false
        }

        // Check for HTTP proxy
        if let httpProxy = proxySettings["HTTPProxy"] as? String, !httpProxy.isEmpty {
            return true
        }

        // Check for HTTPS proxy
        if let httpsProxy = proxySettings["HTTPSProxy"] as? String, !httpsProxy.isEmpty {
            return true
        }

        return false
    }
}
#endif
```

---

## 10. Implementation Recommendations for MYND

### 10.1 Security Priorities by Phase

**MVP (v1.0)**:
1. Keychain storage for API keys (CRITICAL)
2. HTTPS-only with TLS 1.3
3. Local SwiftData encryption (device-level)
4. Basic consent management
5. Data export functionality

**v1.5**:
1. Client-side encryption for CloudKit
2. Multi-device key sync
3. Biometric protection for sensitive operations
4. Full audit logging
5. Anonymization before AI API calls

**v2.0**:
1. Zero-knowledge architecture
2. Secure Enclave key storage
3. Certificate pinning
4. Third-party security audit
5. SOC 2 compliance (if B2B features)

### 10.2 Quick Wins (Implement Immediately)

1. **Never hardcode API keys** - Use Keychain from day one
2. **Set Data Protection** - Enable "Complete" protection on SwiftData store
3. **Enforce HTTPS** - Configure ATS in Info.plist
4. **Minimal Analytics** - Only collect essential, anonymous data
5. **Clear Privacy Policy** - Publish before App Store submission

### 10.3 Key Files to Create

| File | Purpose |
|------|---------|
| `Security/EncryptionService.swift` | AES-256-GCM encryption |
| `Security/KeychainService.swift` | Secure credential storage |
| `Security/BiometricService.swift` | Face ID / Touch ID |
| `Privacy/ConsentManager.swift` | GDPR consent tracking |
| `Privacy/DataExportService.swift` | Data portability |
| `Privacy/DataDeletionService.swift` | Right to erasure |
| `Audit/AuditLogger.swift` | Activity logging |
| `Network/CertificatePinning.swift` | MITM protection |

---

## 11. References

- [Apple CryptoKit Documentation](https://developer.apple.com/documentation/cryptokit)
- [Apple Keychain Services](https://developer.apple.com/documentation/security/keychain_services)
- [Apple Data Protection](https://developer.apple.com/documentation/uikit/protecting_the_user_s_privacy/encrypting_your_app_s_files)
- [CloudKit Security](https://developer.apple.com/documentation/cloudkit)
- [OWASP Mobile Security Testing Guide](https://owasp.org/www-project-mobile-security-testing-guide/)
- [GDPR Text](https://gdpr-info.eu/)
- [Anthropic Privacy Policy](https://www.anthropic.com/privacy)
- [Apple App Store Privacy Guidelines](https://developer.apple.com/app-store/app-privacy-details/)

---

**Document Status**: RESEARCH COMPLETE
**Next Steps**: Implement security components starting with Phase 1 priorities
