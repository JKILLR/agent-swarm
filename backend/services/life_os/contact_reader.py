"""macOS Contacts reader for Life OS integration.

Reads contacts from macOS AddressBook SQLite database.
Requires Full Disk Access permission for the running process.
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional
import os
import threading
import time


class ContactReaderError(Exception):
    """Base exception for contact reader errors."""
    pass


class DatabaseNotFoundError(ContactReaderError):
    """Raised when AddressBook database is not found."""
    pass


class DatabaseAccessError(ContactReaderError):
    """Raised when database cannot be accessed (permissions)."""
    pass


class ContactReader:
    """Read-only access to macOS Contacts database with caching."""

    # Cache TTL in seconds (contacts don't change often)
    CACHE_TTL = 300  # 5 minutes

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the contact reader.

        Args:
            db_path: Path to AddressBook database. Defaults to standard location.
        """
        if db_path is None:
            db_path = os.path.expanduser(
                "~/Library/Application Support/AddressBook/AddressBook-v22.abcddb"
            )

        self.db_path = Path(db_path)
        self._cache: Optional[list[dict]] = None
        self._cache_timestamp: float = 0
        self._cache_lock = threading.Lock()
        self._phone_lookup: dict[str, str] = {}  # normalized_phone -> contact_id
        self._email_lookup: dict[str, str] = {}  # normalized_email -> contact_id
        self._contact_by_id: dict[str, dict] = {}  # contact_id -> contact dict

        self._validate_database()

    def _validate_database(self) -> None:
        """Validate database exists and is accessible."""
        if not self.db_path.exists():
            raise DatabaseNotFoundError(
                f"Contacts database not found at {self.db_path}. "
                "Ensure Contacts app is set up on this Mac."
            )

        try:
            conn = self._get_connection()
            conn.close()
        except sqlite3.OperationalError as e:
            raise DatabaseAccessError(
                f"Cannot access Contacts database: {e}. "
                "Ensure Full Disk Access is granted in System Settings > "
                "Privacy & Security > Full Disk Access."
            )

    def _get_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection."""
        uri = f"file:{self.db_path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to digits only for comparison."""
        digits = re.sub(r'\D', '', phone)
        # Handle US numbers - keep last 10 digits if longer
        if len(digits) > 10 and digits.startswith('1'):
            digits = digits[1:]
        return digits[-10:] if len(digits) >= 10 else digits

    def _normalize_email(self, email: str) -> str:
        """Normalize email for comparison."""
        return email.lower().strip()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (
            self._cache is not None and
            (time.time() - self._cache_timestamp) < self.CACHE_TTL
        )

    def _load_contacts(self) -> list[dict]:
        """Load all contacts from database."""
        contacts_by_pk: dict[int, dict] = {}

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get all person records (not groups)
            cursor.execute("""
                SELECT
                    Z_PK,
                    ZUNIQUEID,
                    ZFIRSTNAME,
                    ZLASTNAME,
                    ZMIDDLENAME,
                    ZNICKNAME,
                    ZORGANIZATION
                FROM ZABCDRECORD
                WHERE ZFIRSTNAME IS NOT NULL
                   OR ZLASTNAME IS NOT NULL
                   OR ZORGANIZATION IS NOT NULL
            """)

            for row in cursor:
                pk = row["Z_PK"]
                # Build display name
                parts = []
                if row["ZFIRSTNAME"]:
                    parts.append(row["ZFIRSTNAME"])
                if row["ZMIDDLENAME"]:
                    parts.append(row["ZMIDDLENAME"])
                if row["ZLASTNAME"]:
                    parts.append(row["ZLASTNAME"])

                name = " ".join(parts) if parts else row["ZORGANIZATION"] or ""

                contacts_by_pk[pk] = {
                    "id": row["ZUNIQUEID"] or str(pk),
                    "name": name,
                    "first_name": row["ZFIRSTNAME"] or "",
                    "last_name": row["ZLASTNAME"] or "",
                    "organization": row["ZORGANIZATION"] or "",
                    "nickname": row["ZNICKNAME"] or "",
                    "phones": [],
                    "emails": [],
                }

            # Get phone numbers
            cursor.execute("""
                SELECT ZOWNER, ZFULLNUMBER, ZLABEL
                FROM ZABCDPHONENUMBER
                WHERE ZFULLNUMBER IS NOT NULL
            """)

            for row in cursor:
                owner = row["ZOWNER"]
                if owner in contacts_by_pk:
                    contacts_by_pk[owner]["phones"].append({
                        "number": row["ZFULLNUMBER"],
                        "label": row["ZLABEL"] or "other",
                    })

            # Get email addresses
            cursor.execute("""
                SELECT ZOWNER, ZADDRESS, ZLABEL
                FROM ZABCDEMAILADDRESS
                WHERE ZADDRESS IS NOT NULL
            """)

            for row in cursor:
                owner = row["ZOWNER"]
                if owner in contacts_by_pk:
                    contacts_by_pk[owner]["emails"].append({
                        "address": row["ZADDRESS"],
                        "label": row["ZLABEL"] or "other",
                    })

            cursor.close()
            conn.close()

        except sqlite3.Error as e:
            raise ContactReaderError(f"Database query failed: {e}")

        return list(contacts_by_pk.values())

    def _refresh_cache(self) -> None:
        """Refresh the contact cache and lookup indexes."""
        with self._cache_lock:
            if self._is_cache_valid():
                return  # Another thread already refreshed

            contacts = self._load_contacts()

            # Build lookup indexes
            phone_lookup: dict[str, str] = {}
            email_lookup: dict[str, str] = {}
            contact_by_id: dict[str, dict] = {}

            for contact in contacts:
                contact_id = contact["id"]
                contact_by_id[contact_id] = contact

                for phone in contact["phones"]:
                    normalized = self._normalize_phone(phone["number"])
                    if normalized:
                        phone_lookup[normalized] = contact_id

                for email in contact["emails"]:
                    normalized = self._normalize_email(email["address"])
                    if normalized:
                        email_lookup[normalized] = contact_id

            self._cache = contacts
            self._cache_timestamp = time.time()
            self._phone_lookup = phone_lookup
            self._email_lookup = email_lookup
            self._contact_by_id = contact_by_id

    def invalidate_cache(self) -> None:
        """Force cache refresh on next access."""
        with self._cache_lock:
            self._cache = None
            self._cache_timestamp = 0

    def get_all_contacts(self) -> list[dict]:
        """Get all contacts.

        Returns:
            List of contact dicts with id, name, phones, emails
        """
        if not self._is_cache_valid():
            self._refresh_cache()
        return self._cache or []

    def search_contacts(self, query: str, limit: int = 20) -> list[dict]:
        """Search contacts by name, email, or phone.

        Args:
            query: Search string (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching contact dicts
        """
        if not self._is_cache_valid():
            self._refresh_cache()

        if not query:
            return []

        query_lower = query.lower()
        query_digits = re.sub(r'\D', '', query)
        results = []

        for contact in self._cache or []:
            if len(results) >= limit:
                break

            # Search by name
            if query_lower in contact["name"].lower():
                results.append(contact)
                continue

            # Search by nickname
            if contact["nickname"] and query_lower in contact["nickname"].lower():
                results.append(contact)
                continue

            # Search by organization
            if contact["organization"] and query_lower in contact["organization"].lower():
                results.append(contact)
                continue

            # Search by email
            for email in contact["emails"]:
                if query_lower in email["address"].lower():
                    results.append(contact)
                    break
            else:
                # Search by phone (if query has digits)
                if query_digits:
                    for phone in contact["phones"]:
                        phone_digits = re.sub(r'\D', '', phone["number"])
                        if query_digits in phone_digits:
                            results.append(contact)
                            break

        return results

    def resolve_handle(self, phone_or_email: str) -> Optional[str]:
        """Resolve a phone number or email to a contact name.

        Args:
            phone_or_email: Phone number or email address

        Returns:
            Contact name if found, None otherwise
        """
        if not self._is_cache_valid():
            self._refresh_cache()

        if not phone_or_email:
            return None

        # Check if it looks like an email
        if "@" in phone_or_email:
            normalized = self._normalize_email(phone_or_email)
            contact_id = self._email_lookup.get(normalized)
        else:
            normalized = self._normalize_phone(phone_or_email)
            contact_id = self._phone_lookup.get(normalized)

        if contact_id:
            contact = self._contact_by_id.get(contact_id)
            if contact:
                return contact["name"]

        return None

    def get_contact_by_id(self, contact_id: str) -> Optional[dict]:
        """Get a contact by ID.

        Args:
            contact_id: The contact's unique ID

        Returns:
            Contact dict or None if not found
        """
        if not self._is_cache_valid():
            self._refresh_cache()
        return self._contact_by_id.get(contact_id)


# Singleton instance
_reader: Optional[ContactReader] = None


def get_contact_reader() -> ContactReader:
    """Get or create the singleton ContactReader instance."""
    global _reader
    if _reader is None:
        _reader = ContactReader()
    return _reader


# Module-level convenience functions
def get_all_contacts() -> list[dict]:
    """Get all contacts."""
    return get_contact_reader().get_all_contacts()


def search_contacts(query: str, limit: int = 20) -> list[dict]:
    """Search contacts by name, email, or phone."""
    return get_contact_reader().search_contacts(query, limit)


def resolve_handle(phone_or_email: str) -> Optional[str]:
    """Resolve a phone number or email to a contact name."""
    return get_contact_reader().resolve_handle(phone_or_email)
