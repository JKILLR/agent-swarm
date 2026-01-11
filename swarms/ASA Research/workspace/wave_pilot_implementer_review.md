# Wave Function Pilot: Implementation Review
## Technical Analysis for Adding Dynamics to Static Wave Model

**Agent**: Implementer Review
**Date**: 2026-01-05
**Input**: `asa_wave_pilot.py` context + DPP experiment requirements
**Objective**: Concrete code architecture and modifications for Phase 1 experiments

---

## EXECUTIVE SUMMARY

The `asa_wave_pilot.py` provides an excellent **static foundation** for the Dynamics Primacy Protocol (DPP) experiments:

| Current State | What It Provides | What's Missing |
|---------------|------------------|----------------|
| Wave functions | Token-basis amplitude mapping | Temporal evolution |
| 21 relational bases | Semantic space structure | Basis activation dynamics |
| Compatibility matrix | Static token-token attention | Dynamic attention flow |
| 100-word vocabulary | Test vocabulary | Usage history tracking |

**Core Finding**: The pilot is a perfect "Layer 4 Topological Landscape" (as Gemini noted) upon which to build dynamics. The architecture needs **three additions**, not replacement:

1. **Temporal Evolution Engine** - wave function propagation
2. **State History Tracking** - trajectory through semantic space
3. **Lesion Infrastructure** - freeze/corrupt operations for DPP

---

## PART 1: CURRENT ARCHITECTURE ANALYSIS

### 1.1 What `asa_wave_pilot.py` Computes

```python
# Core computation (inferred from context)
# Wave function: ψ_token = Σ α_r · φ_r (over 21 bases)
# Compatibility: ⟨ψ_i | ψ_j⟩ = Σ_r α_i(r) · α_j(r)

# Current structure:
wave_matrix: np.ndarray  # Shape: (100 tokens, 21 bases)
compatibility_matrix: np.ndarray  # Shape: (100, 100) - token-token
```

### 1.2 Strengths for DPP Implementation

1. **Natural sparsity** - incompatible tokens have zero overlap by construction
2. **Linguistic grounding** - bases map to real syntactic/semantic relations
3. **Interpretable amplitudes** - can trace why tokens are compatible
4. **Well-defined inner product** - compatibility is just ⟨ψ_i|ψ_j⟩

### 1.3 Limitations for DPP

1. **No time dimension** - wave functions are fixed, not evolving
2. **No activation flow** - compatibility computed but not propagated
3. **No context dependency** - ψ("dog") is same regardless of context
4. **No history** - no tracking of concept usage over time
5. **No noise/temperature** - system is deterministic

---

## PART 2: DYNAMICS MODIFICATIONS REQUIRED

### 2.1 Core Insight: Wave Function as INITIAL CONDITION

The current wave functions should be reinterpreted as **rest states** or **intrinsic affinities**:

```
Current interpretation:   ψ_token = constant definition
Required interpretation:  ψ_token(t=0) = initial condition
                         ψ_token(t) = f(ψ(t-1), context, noise, dynamics)
```

### 2.2 Temporal Evolution Architecture

```python
class DynamicWaveFunction:
    """
    Extends static wave functions with temporal dynamics.
    Core modification to asa_wave_pilot.py framework.
    """

    def __init__(self, static_pilot):
        # Import static infrastructure
        self.bases = static_pilot.bases  # 21 relational bases
        self.vocab = static_pilot.vocab  # 100 words
        self.base_wave = static_pilot.wave_matrix.copy()  # Rest states

        # === NEW: Dynamic state ===
        self.active_wave = self.base_wave.copy()  # Current wave functions
        self.activation = np.zeros(100)  # Token activation levels
        self.temperature = 0.1  # Noise parameter (β)
        self.time_step = 0

        # === NEW: History tracking ===
        self.trajectory = []  # List of (time, wave_state, activation) tuples
        self.usage_counts = np.zeros(100)  # How often each token used

    def evolve(self, context_tokens: List[int], dt: float = 1.0) -> None:
        """
        Single time step of wave function evolution.
        This is the HEARTBEAT of the semantic system.
        """
        # Step 1: Context activation
        context_activation = np.zeros(100)
        for tok in context_tokens:
            context_activation[tok] = 1.0

        # Step 2: Propagate through compatibility
        # Tokens similar to active tokens get activated
        propagated = self.compatibility_matrix @ context_activation
        propagated = propagated / (np.max(propagated) + 1e-8)  # Normalize

        # Step 3: Update activation with noise
        noise = np.random.normal(0, self.temperature, 100)
        self.activation = 0.9 * self.activation + 0.1 * propagated + noise
        self.activation = np.clip(self.activation, 0, 1)

        # Step 4: Wave function adaptation (key dynamic!)
        # Active tokens strengthen their wave components
        for i, act in enumerate(self.activation):
            if act > 0.5:
                # Strengthen connections to active basis components
                active_bases = np.where(self.active_wave[i] > 0.1)[0]
                for b in active_bases:
                    self.active_wave[i, b] *= (1 + 0.01 * act)

        # Renormalize wave functions
        norms = np.linalg.norm(self.active_wave, axis=1, keepdims=True)
        self.active_wave = self.active_wave / (norms + 1e-8)

        # Step 5: Record trajectory
        self.trajectory.append({
            'time': self.time_step,
            'wave_snapshot': self.active_wave.copy(),
            'activation': self.activation.copy()
        })
        self.time_step += 1

    @property
    def compatibility_matrix(self) -> np.ndarray:
        """Dynamic compatibility - changes as waves evolve."""
        return self.active_wave @ self.active_wave.T
```

**Estimated LOC**: 150 for core DynamicWaveFunction

### 2.3 Heartbeat Implementation

```python
class SemanticHeartbeat:
    """
    Background dynamics that run even without input.
    Tests the "meaning as ongoing process" hypothesis.
    """

    def __init__(self, wave_system: DynamicWaveFunction):
        self.system = wave_system
        self.heartbeat_hz = 10  # Beats per second
        self.running = False
        self._thread = None

    def start(self):
        """Begin continuous semantic activity."""
        self.running = True
        self._thread = threading.Thread(target=self._heartbeat_loop)
        self._thread.start()

    def _heartbeat_loop(self):
        """The semantic heartbeat - runs continuously."""
        while self.running:
            # Spontaneous activation: random token gets small activation
            spontaneous = np.random.choice(100)
            self.system.activation[spontaneous] += 0.1 * np.random.rand()

            # Evolve with no external context (internal dynamics only)
            self.system.evolve(context_tokens=[], dt=1/self.heartbeat_hz)

            # Check for criticality (self-tuning)
            self._tune_criticality()

            time.sleep(1 / self.heartbeat_hz)

    def _tune_criticality(self):
        """Self-organize toward critical regime."""
        # Measure activation variance
        var = np.var(self.system.activation)

        # At criticality, variance should be intermediate
        if var < 0.01:  # Too ordered
            self.system.temperature *= 1.1
        elif var > 0.25:  # Too chaotic
            self.system.temperature *= 0.9

    def stop(self):
        """Halt the heartbeat."""
        self.running = False
        if self._thread:
            self._thread.join()
```

**Estimated LOC**: 80 for SemanticHeartbeat

---

## PART 3: DPP CONDITIONS ARCHITECTURE

### 3.1 The Core DPP Test Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DYNAMICS PRIMACY PROTOCOL                         │
│                                                                      │
│   CONDITION 1: DYNAMICS LESION                                       │
│   ─────────────────────────────                                      │
│   • Stop heartbeat                                                   │
│   • Freeze wave evolution                                            │
│   • PRESERVE: All wave amplitudes, compatibility matrix              │
│   • Test: Can system still do semantic tasks?                        │
│                                                                      │
│   CONDITION 2: STRUCTURE LESION                                      │
│   ─────────────────────────────                                      │
│   • Keep heartbeat running                                           │
│   • ADD NOISE to wave amplitudes                                     │
│   • PRESERVE: Dynamics engine, evolution rules                       │
│   • Test: Can dynamics compensate for corrupted structure?           │
│                                                                      │
│   KEY PREDICTION:                                                    │
│   If meaning is a VERB: Dynamics lesion kills performance            │
│   If meaning is a NOUN: Structure lesion kills performance           │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Lesion Implementation

```python
class DPPLesionController:
    """
    Implements the Dynamics Primacy Protocol lesion conditions.
    Central to the experimental design.
    """

    def __init__(self, system: DynamicWaveFunction, heartbeat: SemanticHeartbeat):
        self.system = system
        self.heartbeat = heartbeat
        self._frozen_state = None

    def dynamics_lesion(self) -> 'FrozenSystem':
        """
        CONDITION 1: Freeze dynamics, preserve structure.
        This tests if understanding survives when process stops.
        """
        # Stop all dynamics
        self.heartbeat.stop()

        # Capture and freeze state
        self._frozen_state = {
            'wave_matrix': self.system.active_wave.copy(),
            'activation': self.system.activation.copy(),
            'compatibility': self.system.compatibility_matrix.copy()
        }

        # Return frozen system wrapper
        return FrozenSystem(self._frozen_state)

    def structure_lesion(self, corruption_level: float = 0.3) -> DynamicWaveFunction:
        """
        CONDITION 2: Corrupt structure, preserve dynamics.
        This tests if dynamics can compensate for structural damage.
        """
        corrupted = copy.deepcopy(self.system)

        # Corrupt wave amplitudes
        n_corrupt = int(100 * 21 * corruption_level)
        indices = np.random.choice(100 * 21, n_corrupt, replace=False)
        flat = corrupted.active_wave.flatten()
        for idx in indices:
            flat[idx] += np.random.normal(0, 0.5)
        corrupted.active_wave = flat.reshape(100, 21)

        # Renormalize (corrupted but valid wave functions)
        norms = np.linalg.norm(corrupted.active_wave, axis=1, keepdims=True)
        corrupted.active_wave = corrupted.active_wave / (norms + 1e-8)

        # DYNAMICS CONTINUE - heartbeat keeps running!
        return corrupted

    def restore(self) -> DynamicWaveFunction:
        """Restore system from frozen state."""
        if self._frozen_state:
            self.system.active_wave = self._frozen_state['wave_matrix'].copy()
            self.system.activation = self._frozen_state['activation'].copy()
            self.heartbeat.start()
        return self.system


class FrozenSystem:
    """
    A frozen semantic system - all structure, no dynamics.
    Used for dynamics lesion condition.
    """

    def __init__(self, frozen_state: Dict):
        self.wave_matrix = frozen_state['wave_matrix']
        self.activation = frozen_state['activation']  # Fixed, never changes
        self.compatibility = frozen_state['compatibility']
        self._has_dynamics = False

    def query_compatibility(self, token_i: int, token_j: int) -> float:
        """Can still do static lookups."""
        return self.compatibility[token_i, token_j]

    def evolve(self, *args, **kwargs):
        """Frozen systems don't evolve."""
        pass  # No-op - this is the point of the lesion

    def process_context(self, context: List[int]) -> np.ndarray:
        """
        Process input using frozen structure.
        No dynamic propagation - just static compatibility lookup.
        """
        # Can only do static operations
        response = np.zeros(100)
        for tok in context:
            response += self.compatibility[tok]
        return response / (len(context) + 1e-8)
```

**Estimated LOC**: 120 for DPPLesionController + FrozenSystem

### 3.3 Confusion Thermodynamics (Gemini's Suggestion)

```python
class ConfusionThermometer:
    """
    Measures entropy fluctuations in wave function overlaps.
    Gemini's key insight: confusion = phase transition signature.
    """

    def measure_confusion_signature(
        self,
        system: Union[DynamicWaveFunction, FrozenSystem],
        probe_token: int,
        repetitions: int = 50
    ) -> Dict:
        """
        Induce semantic satiation and measure response.

        Living system: Should show entropy fluctuations (phase transition)
        Zombie/Frozen: Should show no signal change
        """
        entropy_history = []
        activation_history = []

        for rep in range(repetitions):
            # Expose to same token repeatedly
            if hasattr(system, 'evolve'):
                system.evolve(context_tokens=[probe_token])

            # Measure entropy of activation distribution
            act = system.activation if hasattr(system, 'activation') else np.zeros(100)
            # Add small epsilon to avoid log(0)
            p = (np.abs(act) + 1e-10) / (np.sum(np.abs(act)) + 1e-8)
            entropy = -np.sum(p * np.log(p))
            entropy_history.append(entropy)
            activation_history.append(act[probe_token])

        return {
            'entropy_history': np.array(entropy_history),
            'activation_history': np.array(activation_history),
            'entropy_variance': np.var(entropy_history),
            'entropy_mean': np.mean(entropy_history),
            'shows_fluctuation': np.var(entropy_history) > 0.01,
            'satiation_detected': activation_history[-1] < activation_history[0] * 0.5
        }
```

**Estimated LOC**: 60 for ConfusionThermometer

---

## PART 4: PHASE 1 EXPERIMENT IMPLEMENTATIONS

### 4.1 Experiment 1.1: Criticality-Correctness Correlation

```python
class CriticalityCorrectnessExperiment:
    """
    Phase 1.1: Does correctness emerge from criticality?
    Maps wave function dynamics to semantic task performance.
    """

    def __init__(self, wave_system: DynamicWaveFunction):
        self.system = wave_system
        self.thermometer = CriticalityThermometer()

    def measure_criticality(self) -> float:
        """
        Measure system's criticality via activation avalanches.
        At criticality: power law exponent ~1.5
        """
        # Collect activation avalanches
        avalanches = []
        for _ in range(100):
            # Perturb single token
            perturbed = np.random.choice(100)
            initial_act = self.system.activation.copy()
            self.system.activation[perturbed] += 0.5

            # Let propagate
            total_spread = 0
            for step in range(20):
                self.system.evolve([])
                spread = np.sum(np.abs(self.system.activation - initial_act) > 0.01)
                total_spread += spread

            avalanches.append(total_spread)

        # Fit power law
        try:
            fit = powerlaw.Fit(avalanches)
            return fit.alpha  # Should be ~1.5 at criticality
        except:
            return 0.0

    def measure_correctness(self, test_pairs: List[Tuple]) -> float:
        """
        Measure semantic task performance.
        test_pairs: [(token_a, token_b, should_be_similar)]
        """
        correct = 0
        for tok_a, tok_b, expected in test_pairs:
            similarity = self.system.compatibility_matrix[tok_a, tok_b]
            predicted = similarity > 0.5
            if predicted == expected:
                correct += 1
        return correct / len(test_pairs)

    def run(self, n_samples: int = 100) -> Dict:
        """Run the criticality-correctness correlation experiment."""
        results = []

        for _ in range(n_samples):
            # Vary temperature to explore different regimes
            self.system.temperature = np.random.uniform(0.01, 0.5)

            # Let system settle
            for _ in range(50):
                self.system.evolve([])

            criticality = self.measure_criticality()
            correctness = self.measure_correctness(self._get_test_pairs())

            results.append({
                'temperature': self.system.temperature,
                'criticality': criticality,
                'correctness': correctness
            })

        # Compute correlation
        crits = [r['criticality'] for r in results]
        corrs = [r['correctness'] for r in results]
        correlation, p_value = pearsonr(crits, corrs)

        return {
            'results': results,
            'correlation': correlation,
            'p_value': p_value,
            'falsified': correlation < 0.3,
            'supported': correlation > 0.5 and p_value < 0.01
        }
```

**Estimated LOC**: 100

### 4.2 Experiment 1.2: Zombie Baseline Construction

```python
class ZombieBaseline:
    """
    Phase 1.2: Build lookup table zombie from dynamic system.
    The zombie has identical I/O but no dynamics.
    """

    def __init__(self, source_system: DynamicWaveFunction):
        self.lookup_table = {}
        self._build_table(source_system)

    def _build_table(self, source: DynamicWaveFunction, n_samples: int = 10000):
        """Exhaustively sample source system's responses."""
        for _ in range(n_samples):
            # Random context
            context_size = np.random.randint(1, 6)
            context = np.random.choice(100, context_size, replace=False).tolist()

            # Get source response
            source.evolve(context)
            response = source.activation.copy()

            # Store in lookup table
            key = tuple(sorted(context))
            if key not in self.lookup_table:
                self.lookup_table[key] = response

    def process(self, context: List[int]) -> np.ndarray:
        """Lookup-only processing."""
        key = tuple(sorted(context))
        if key in self.lookup_table:
            return self.lookup_table[key]
        else:
            # Approximate via nearest neighbor
            best_match = None
            best_overlap = -1
            for stored_key in self.lookup_table:
                overlap = len(set(stored_key) & set(context))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = stored_key
            return self.lookup_table.get(best_match, np.zeros(100))

    def evolve(self, *args, **kwargs):
        """Zombies don't evolve."""
        pass

    @property
    def resting_activity(self) -> float:
        """Zombies have no resting activity."""
        return 0.0


class ZombieBaselineExperiment:
    """Validates that zombie matches living system on training distribution."""

    def run(self, living: DynamicWaveFunction, zombie: ZombieBaseline) -> Dict:
        match_count = 0
        total = 1000

        for _ in range(total):
            context = np.random.choice(100, 3, replace=False).tolist()

            living.evolve(context)
            living_response = living.activation.copy()

            zombie_response = zombie.process(context)

            # Check if responses match (within tolerance)
            similarity = np.corrcoef(living_response, zombie_response)[0, 1]
            if similarity > 0.9:
                match_count += 1

        return {
            'match_rate': match_count / total,
            'zombie_valid': match_count / total > 0.95,
            'zombie_table_size': len(zombie.lookup_table)
        }
```

**Estimated LOC**: 100

### 4.3 Experiment 1.3: Semantic Heartbeat Detection

```python
class SemanticHeartbeatExperiment:
    """
    Phase 1.3: Do immanent systems show structured resting activity?
    The key test of "meaning as ongoing process".
    """

    def run(
        self,
        living: DynamicWaveFunction,
        frozen: FrozenSystem,
        zombie: ZombieBaseline,
        observation_time: int = 1000
    ) -> Dict:
        """
        Measure basal semantic rate (BSR) across conditions.
        """
        # Measure living system
        living_activity = self._measure_resting(living, observation_time)

        # Measure frozen system
        frozen_activity = self._measure_resting(frozen, observation_time)

        # Measure zombie
        zombie_activity = self._measure_resting(zombie, observation_time)

        return {
            'living_bsr': living_activity['mean_rate'],
            'living_structure': living_activity['spectral_peaks'],
            'frozen_bsr': frozen_activity['mean_rate'],  # Should be 0
            'zombie_bsr': zombie_activity['mean_rate'],  # Should be 0
            'heartbeat_detected': living_activity['mean_rate'] > 0.01,
            'has_spectral_structure': len(living_activity['spectral_peaks']) > 0
        }

    def _measure_resting(self, system, duration: int) -> Dict:
        """Measure resting activity over time."""
        activity_trace = []

        for t in range(duration):
            if hasattr(system, 'evolve'):
                system.evolve([])  # No input

            # Measure total activation
            if hasattr(system, 'activation'):
                activity = np.sum(np.abs(system.activation))
            else:
                activity = 0.0
            activity_trace.append(activity)

        # Spectral analysis
        activity_trace = np.array(activity_trace)
        fft = np.fft.fft(activity_trace)
        freqs = np.fft.fftfreq(len(activity_trace))

        # Find peaks
        power = np.abs(fft) ** 2
        peaks = []
        for i in range(1, len(power) - 1):
            if power[i] > power[i-1] and power[i] > power[i+1]:
                if power[i] > np.mean(power) * 2:
                    peaks.append(freqs[i])

        return {
            'mean_rate': np.mean(activity_trace),
            'variance': np.var(activity_trace),
            'spectral_peaks': peaks
        }
```

**Estimated LOC**: 80

### 4.4 Experiment 1.4: Topology-Semantics Isomorphism

```python
class TopologySemanticIsomorphismExperiment:
    """
    Phase 1.4: Do identical wave function structures produce identical semantics?
    Tests whether structure = semantics.
    """

    def run(self, n_tests: int = 500) -> Dict:
        # Create three systems with IDENTICAL wave structures
        base_wave = np.random.randn(100, 21)
        base_wave = base_wave / np.linalg.norm(base_wave, axis=1, keepdims=True)

        # Different dynamics implementations
        system_a = DynamicWaveFunction.__new__(DynamicWaveFunction)
        system_a.active_wave = base_wave.copy()
        system_a.temperature = 0.1  # Low noise
        system_a.activation = np.zeros(100)

        system_b = DynamicWaveFunction.__new__(DynamicWaveFunction)
        system_b.active_wave = base_wave.copy()
        system_b.temperature = 0.2  # Medium noise
        system_b.activation = np.zeros(100)

        system_c = DynamicWaveFunction.__new__(DynamicWaveFunction)
        system_c.active_wave = base_wave.copy()
        system_c.temperature = 0.05  # Very low noise
        system_c.activation = np.zeros(100)

        # Run same test battery on all three
        errors_a = self._collect_errors(system_a, n_tests)
        errors_b = self._collect_errors(system_b, n_tests)
        errors_c = self._collect_errors(system_c, n_tests)

        # Compute error pattern correlations
        corr_ab = np.corrcoef(errors_a, errors_b)[0, 1]
        corr_ac = np.corrcoef(errors_a, errors_c)[0, 1]
        corr_bc = np.corrcoef(errors_b, errors_c)[0, 1]

        return {
            'correlation_ab': corr_ab,
            'correlation_ac': corr_ac,
            'correlation_bc': corr_bc,
            'mean_correlation': np.mean([corr_ab, corr_ac, corr_bc]),
            'topology_determines_semantics': min(corr_ab, corr_ac, corr_bc) > 0.8,
            'falsified': min(corr_ab, corr_ac, corr_bc) < 0.5
        }

    def _collect_errors(self, system, n_tests: int) -> np.ndarray:
        """Collect error pattern on semantic tasks."""
        errors = []
        for _ in range(n_tests):
            # Random similarity judgment
            tok_a, tok_b = np.random.choice(100, 2, replace=False)
            compatibility = system.active_wave[tok_a] @ system.active_wave[tok_b]

            # Ground truth: based on shared bases
            shared_bases = np.sum((system.active_wave[tok_a] > 0.1) &
                                  (system.active_wave[tok_b] > 0.1))
            ground_truth = shared_bases > 2

            # System prediction
            predicted = compatibility > 0.3

            errors.append(1 if predicted != ground_truth else 0)

        return np.array(errors)
```

**Estimated LOC**: 80

---

## PART 5: MISSING INFRASTRUCTURE

### 5.1 Critical Missing Components

| Component | Purpose | Priority | Est. LOC |
|-----------|---------|----------|----------|
| **CriticalityThermometer** | Measure avalanche distributions, exponents | P0 | 200 |
| **PhaseSpaceMapper** | Track (alpha, beta, history) coordinates | P1 | 150 |
| **SemanticCalorimeter** | Measure "energy" consumption per operation | P1 | 150 |
| **BenchmarkSuite** | Standardized semantic tasks | P0 | 200 |
| **TrajectoryRecorder** | Log concept evolution over time | P1 | 100 |
| **StatisticalAnalyzer** | Correlation, power law fitting, significance | P0 | 150 |

### 5.2 CriticalityThermometer Implementation

```python
class CriticalityThermometer:
    """
    Measures whether system operates at semantic criticality.
    Core measurement infrastructure for all experiments.
    """

    def measure_avalanche_distribution(
        self,
        system: DynamicWaveFunction,
        n_perturbations: int = 200
    ) -> Dict:
        """
        Perturb system, measure cascade sizes.
        At criticality: power law with exponent ~1.5
        """
        avalanche_sizes = []

        for _ in range(n_perturbations):
            # Save state
            initial_state = system.activation.copy()

            # Small perturbation to random token
            perturbed = np.random.choice(100)
            system.activation[perturbed] += 0.3

            # Measure cascade
            cascade_size = 0
            for step in range(50):
                system.evolve([])
                changed = np.sum(np.abs(system.activation - initial_state) > 0.05)
                cascade_size += changed
                if changed == 0:
                    break

            avalanche_sizes.append(cascade_size)

            # Reset
            system.activation = initial_state

        # Fit power law
        avalanche_sizes = np.array(avalanche_sizes)
        avalanche_sizes = avalanche_sizes[avalanche_sizes > 0]

        try:
            fit = powerlaw.Fit(avalanche_sizes, discrete=True)
            is_power_law = fit.distribution_compare('power_law', 'exponential')[0] > 0

            return {
                'exponent': fit.alpha,
                'xmin': fit.xmin,
                'is_power_law': is_power_law,
                'is_critical': 1.3 < fit.alpha < 1.7 and is_power_law,
                'raw_sizes': avalanche_sizes
            }
        except:
            return {
                'exponent': None,
                'is_critical': False,
                'raw_sizes': avalanche_sizes
            }

    def criticality_score(self, system: DynamicWaveFunction) -> float:
        """
        Combined criticality score in [0, 1].
        0.5-0.7 is the "meaning zone".
        """
        result = self.measure_avalanche_distribution(system)

        if result['exponent'] is None:
            return 0.0

        # Score based on distance from ideal exponent (1.5)
        exponent_score = 1.0 - min(abs(result['exponent'] - 1.5), 1.0)

        # Power law quality score
        pl_score = 1.0 if result['is_power_law'] else 0.0

        return 0.7 * exponent_score + 0.3 * pl_score
```

**Estimated LOC**: 100

### 5.3 BenchmarkSuite Implementation

```python
class BenchmarkSuite:
    """
    Standardized semantic tasks for comparing systems.
    Maps wave function operations to semantic performance.
    """

    def __init__(self, vocab_size: int = 100, n_bases: int = 21):
        self.vocab_size = vocab_size
        self.n_bases = n_bases
        self._generate_test_cases()

    def _generate_test_cases(self):
        """Generate standardized test cases."""
        # Analogy completion: A:B::C:?
        self.analogies = []
        # In wave function terms: find D such that (ψ_A - ψ_B) ≈ (ψ_C - ψ_D)

        # Similarity judgment
        self.similarity_pairs = []

        # Contradiction detection
        self.contradictions = []

    def analogy_test(
        self,
        system: Union[DynamicWaveFunction, FrozenSystem, ZombieBaseline],
        n_tests: int = 50
    ) -> float:
        """
        A:B::C:? - find token D that completes analogy.
        Uses wave function arithmetic: ψ_D ≈ ψ_C + (ψ_B - ψ_A)
        """
        correct = 0

        for _ in range(n_tests):
            # Generate analogy from known relations
            a, b, c, d = self._generate_analogy()

            if hasattr(system, 'active_wave'):
                wave = system.active_wave
            else:
                wave = system.wave_matrix

            # Predict D via wave arithmetic
            predicted_wave = wave[c] + (wave[b] - wave[a])

            # Find closest actual token
            similarities = wave @ predicted_wave
            predicted_d = np.argmax(similarities)

            if predicted_d == d:
                correct += 1

        return correct / n_tests

    def similarity_judgment(
        self,
        system: Union[DynamicWaveFunction, FrozenSystem, ZombieBaseline],
        n_tests: int = 100
    ) -> float:
        """Which token is more similar to X: Y or Z?"""
        correct = 0

        for _ in range(n_tests):
            x, y, z, answer = self._generate_similarity_triplet()

            if hasattr(system, 'compatibility_matrix'):
                compat = system.compatibility_matrix
            else:
                wave = system.wave_matrix if hasattr(system, 'wave_matrix') else system.active_wave
                compat = wave @ wave.T

            sim_xy = compat[x, y]
            sim_xz = compat[x, z]

            predicted = 'y' if sim_xy > sim_xz else 'z'
            if predicted == answer:
                correct += 1

        return correct / n_tests

    def novel_combination(
        self,
        system: DynamicWaveFunction,
        n_tests: int = 50
    ) -> float:
        """
        Combine concepts A+B, predict emergent properties.
        This requires DYNAMICS - frozen systems should fail.
        """
        correct = 0

        for _ in range(n_tests):
            a, b, expected_properties = self._generate_combination()

            # Activate both concepts
            system.evolve([a, b])

            # Let dynamics settle
            for _ in range(10):
                system.evolve([])

            # Check if expected properties emerge in activation
            emergent_correct = 0
            for prop_token in expected_properties:
                if system.activation[prop_token] > 0.3:
                    emergent_correct += 1

            if emergent_correct >= len(expected_properties) * 0.5:
                correct += 1

        return correct / n_tests

    def full_suite(
        self,
        system: Union[DynamicWaveFunction, FrozenSystem, ZombieBaseline]
    ) -> Dict:
        """Run all benchmarks."""
        return {
            'analogy': self.analogy_test(system),
            'similarity': self.similarity_judgment(system),
            'novel_combination': self.novel_combination(system) if hasattr(system, 'evolve') else 0.0,
            'overall': np.mean([
                self.analogy_test(system),
                self.similarity_judgment(system)
            ])
        }
```

**Estimated LOC**: 150

---

## PART 6: COMPLETE DEPENDENCY GRAPH

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       WAVE PILOT → DPP DEPENDENCY GRAPH                      │
└─────────────────────────────────────────────────────────────────────────────┘

Layer 0: EXISTING (asa_wave_pilot.py)
┌─────────────────────────────────────────────────────────────────────────────┐
│  wave_matrix (100×21)  │  compatibility_matrix  │  vocab  │  bases (21)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Layer 1: DYNAMICS CORE (NEW)
┌─────────────────────────────────────────────────────────────────────────────┐
│  DynamicWaveFunction         │  SemanticHeartbeat      │  TrajectoryRecorder│
│  - active_wave (evolving)    │  - heartbeat_loop()     │  - log_state()     │
│  - evolve()                  │  - spontaneous_act()    │  - get_history()   │
│  - temperature               │  - tune_criticality()   │                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Layer 2: LESION INFRASTRUCTURE (NEW)
┌─────────────────────────────────────────────────────────────────────────────┐
│  DPPLesionController         │  FrozenSystem           │  ZombieBaseline    │
│  - dynamics_lesion()         │  - frozen compatibility │  - lookup_table    │
│  - structure_lesion()        │  - no evolve()          │  - no dynamics     │
│  - restore()                 │                         │                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Layer 3: MEASUREMENT (NEW)
┌─────────────────────────────────────────────────────────────────────────────┐
│  CriticalityThermometer      │  ConfusionThermometer   │  SemanticCalorimeter│
│  - avalanche_dist()          │  - confusion_sig()      │  - energy_per_op() │
│  - criticality_score()       │  - entropy_history      │  - basal_rate()    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Layer 4: EXPERIMENTS
┌─────────────────────────────────────────────────────────────────────────────┐
│  Exp 1.1: Criticality-Correctness  │  Exp 1.2: Zombie Baseline             │
│  Exp 1.3: Semantic Heartbeat       │  Exp 1.4: Topology-Semantics          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Layer 5: PHASE 2+ (FUTURE)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Confusion Thermodynamics  │  Dynamics Lesion Study  │  Use-Topology       │
│  Hermeneutic Circle        │  Historical Criticality │  Bell Test          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PART 7: COMPLEXITY AND LOC ESTIMATES

### 7.1 By Component

| Component | Priority | Est. LOC | Complexity | Dependencies |
|-----------|----------|----------|------------|--------------|
| DynamicWaveFunction | P0 | 150 | Medium | asa_wave_pilot |
| SemanticHeartbeat | P0 | 80 | Medium | DynamicWaveFunction |
| DPPLesionController | P0 | 120 | Low | DynamicWaveFunction, Heartbeat |
| FrozenSystem | P0 | 60 | Low | None |
| ZombieBaseline | P0 | 100 | Medium | DynamicWaveFunction |
| CriticalityThermometer | P0 | 200 | High | powerlaw library |
| ConfusionThermometer | P1 | 60 | Low | DynamicWaveFunction |
| BenchmarkSuite | P0 | 200 | Medium | All systems |
| TrajectoryRecorder | P1 | 100 | Low | DynamicWaveFunction |
| PhaseSpaceMapper | P1 | 150 | Medium | CriticalityThermometer |
| StatisticalAnalyzer | P0 | 150 | Medium | scipy.stats |
| **Phase 1 Experiments** | P0 | 400 | Medium | All above |

### 7.2 Total Estimates

| Category | LOC | Days |
|----------|-----|------|
| Core Dynamics | 330 | 3 |
| Lesion Infrastructure | 280 | 2 |
| Measurement | 460 | 4 |
| Phase 1 Experiments | 400 | 4 |
| Tests + Integration | 300 | 2 |
| **TOTAL** | **~1,800** | **~15** |

### 7.3 Implementation Order

```
Week 1: Core Infrastructure
├── Day 1-2: DynamicWaveFunction + SemanticHeartbeat
├── Day 3: DPPLesionController + FrozenSystem
├── Day 4-5: CriticalityThermometer + StatisticalAnalyzer

Week 2: Experimental Framework
├── Day 6-7: ZombieBaseline + BenchmarkSuite
├── Day 8-9: Phase 1 Experiments (1.1-1.4)
├── Day 10: Integration tests

Week 3: Validation + Phase 1 Runs
├── Day 11-12: Full system validation
├── Day 13-15: Phase 1 experiment runs + analysis
```

---

## PART 8: KEY DESIGN DECISIONS

### 8.1 Wave Function Evolution vs. Graph Propagation

The existing experiments (round3_implementer) describe a graph-based `LivingSemanticSubstrate`. The wave pilot uses a different formalism.

**Recommendation**: Keep wave function formalism for the pilot. The wave function ψ naturally encodes:
- Token identity (the wave)
- Relational capacity (basis amplitudes)
- Compatibility (inner products)

Graph propagation can be derived FROM wave function dynamics:
```python
# Graph adjacency emerges from wave compatibility
adjacency[i,j] = 1 if ⟨ψ_i|ψ_j⟩ > threshold else 0
```

### 8.2 Heartbeat Frequency

**Recommendation**: Start with 10 Hz (not 100 Hz as in round3_implementer).
- Lower frequency = more interpretable
- Can increase if needed
- 10 Hz allows ~1000 timesteps in 100 seconds

### 8.3 Temperature/Noise Parameter

The wave pilot doesn't mention noise. This is critical for:
- Self-organized criticality
- Avoiding frozen states
- Testing the "edge of chaos" hypothesis

**Recommendation**: Add temperature parameter β with self-tuning.

### 8.4 Entropy Fluctuations (Gemini's Insight)

Gemini suggested using ENTROPY FLUCTUATIONS as confusion signature:
- Living system confusion = Phase Transition (entropy spikes)
- Zombie system confusion = Zero Signal (no entropy change)

**Implementation**: ConfusionThermometer tracks entropy of activation distribution over time during semantic satiation.

---

## PART 9: INTEGRATION WITH EXISTING ROUND 3 ARCHITECTURE

The round3_implementer specified a `LivingSemanticSubstrate` class. The wave pilot provides an alternative formulation.

### 9.1 Mapping Between Architectures

| round3_implementer | Wave Pilot Equivalent |
|--------------------|----------------------|
| `graph` (nx.Graph) | Derived from compatibility_matrix thresholding |
| `node_states` (N×64) | `wave_matrix` (100×21) |
| `edge_weights` | `⟨ψ_i\|ψ_j⟩` values |
| `heartbeat()` | `SemanticHeartbeat` class |
| `_spontaneous_activation()` | Random activation + evolve([]) |
| `_hebbian_update()` | Wave amplitude strengthening in `evolve()` |
| `freeze_dynamics()` | `DPPLesionController.dynamics_lesion()` |
| `corrupt_memory()` | `DPPLesionController.structure_lesion()` |

### 9.2 Recommendation: Parallel Implementation

Both architectures test the same DPP hypothesis. Recommend:
1. Implement wave pilot dynamics (this document)
2. Implement graph-based LivingSemanticSubstrate (round3_implementer)
3. Compare results across architectures
4. If both show same DPP effect: architecture-independent validation

---

## CONCLUSION

The `asa_wave_pilot.py` provides an excellent static foundation. To enable DPP experiments:

1. **Add temporal evolution** (DynamicWaveFunction)
2. **Add heartbeat dynamics** (SemanticHeartbeat)
3. **Implement lesion conditions** (DPPLesionController)
4. **Build measurement infrastructure** (Thermometers, Benchmarks)
5. **Run Phase 1 experiments** (4 quick falsification tests)

**Total estimated work**: ~1,800 LOC, ~15 days

The wave function formalism offers interpretable semantics:
- Token meaning = wave function ψ
- Relational capacity = basis amplitudes
- Semantic compatibility = inner product ⟨ψ_i|ψ_j⟩
- Dynamics = wave evolution over time

If Phase 1 experiments show:
- Criticality-correctness correlation > 0.5
- Zombie successfully built (match rate > 0.95)
- Heartbeat detected (BSR > 0.01)
- Topology determines semantics (correlation > 0.8)

Then proceed to Phase 2 (Confusion Thermodynamics, Dynamics Lesion Study).

---

**Status**: IMPLEMENTATION REVIEW COMPLETE
**Date**: 2026-01-05
**Next Step**: Begin DynamicWaveFunction implementation
