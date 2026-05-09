from __future__ import annotations

import numpy as np
import numpy.typing as npt
import qutip as qt

from qubit import HiddenQubit

# Default time resolution for wait() free-evolution segments.
_WAIT_STEPS = 200


class VirtualQubitPair:
    """A two-qubit simulator initialised in the joint ground state |00>.

    Each qubit can be individually driven by a LabOne Q waveform.  The pair is
    coupled via a transverse (XX + YY) exchange interaction of strength J.

    Bell state preparation is the user's responsibility.
    CHSH-optimal angles for the prepared state |Phi-> = (|00> - |11>) / sqrt(2):
      (q0):  a = 0,        a' = pi/2
      (q1):  b = -pi/4,    b' = +pi/4

    Typical experiment flow:

        pair.reset()                              # |00>
        pair.evolve(t_prep, w0, w1, f0, f1)       # drive Bell prep -> |Phi->
        pair.evolve(t_meas, w0, w1, f0, f1)       # rotate into measurement basis
        bits = pair.measure(shots=4000)           # sample

    Args:
        q0:         First qubit.
        q1:         Second qubit.
        J_coupling: Exchange coupling strength [rad/s].
    """

    # Ground state |00> as a density matrix.
    _GROUND_STATE: qt.Qobj = qt.tensor(qt.basis(2, 0), qt.basis(2, 0)).proj()

    def __init__(
        self,
        q0: HiddenQubit,
        q1: HiddenQubit,
        J_coupling: float = 2 * np.pi * 3e6,
    ) -> None:
        self.q0 = q0
        self.q1 = q1
        self._J = J_coupling
        self.state: qt.Qobj = self._GROUND_STATE

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the pair to the joint ground state |00>.

        Call this at the beginning of each new experiment to discard any
        previously accumulated state.  To prepare an entangled state, drive
        the pair through a Bell preparation pulse sequence via `evolve`.
        """
        self.state = self._GROUND_STATE

    # ------------------------------------------------------------------
    # Two-qubit operator constructors
    # These replace the generic "embed" lambdas: every operator that acts
    # on q0 is tensor-producted with I₂ on q1, and vice versa.
    # ------------------------------------------------------------------

    @staticmethod
    def _on_q0(op: qt.Qobj) -> qt.Qobj:
        """Embed a single-qubit operator so that it acts on qubit 0."""
        return qt.tensor(op, qt.qeye(2))

    @staticmethod
    def _on_q1(op: qt.Qobj) -> qt.Qobj:
        """Embed a single-qubit operator so that it acts on qubit 1."""
        return qt.tensor(qt.qeye(2), op)

    # ------------------------------------------------------------------
    # Hamiltonian helpers
    # ------------------------------------------------------------------

    def _drive_hamiltonian(
        self,
        wave_q0: npt.NDArray[np.complex128],
        wave_q1: npt.NDArray[np.complex128],
        drive_freq_q0: float,
        drive_freq_q1: float,
    ) -> list:
        """Assemble the full two-qubit drive Hamiltonian.

        Each qubit contributes its own single-qubit terms (detuning + IQ drive),
        embedded into the two-qubit space via `_on_q0` / `_on_q1`.
        """
        return self.q0.hamiltonian_terms(
            wave_q0, drive_freq_q0, embed=self._on_q0
        ) + self.q1.hamiltonian_terms(wave_q1, drive_freq_q1, embed=self._on_q1)

    def _coupling_hamiltonian(self) -> list:
        """Build the static ZZ coupling H = (J/2) * Z0 * Z1.

        ZZ-type coupling is the natural source of CPHASE gates in
        superconducting hardware (e.g. via the |11>-|02> avoided crossing in
        flux-tunable transmons).  Evolution for time pi/(2J) gives
        exp(-i*pi/4 * Z0*Z1), which combined with single-qubit Rz(-pi/2) on
        each qubit realises a CPHASE gate.

        Importantly, [ZZ, Z_i] = 0, so the gate commutes with per-qubit
        detunings and is unaffected by qubit-frequency mismatches.

        Returns:
            Static QuTiP operator for the coupling Hamiltonian.
        """
        ZZ = self._on_q0(qt.sigmaz()) * self._on_q1(qt.sigmaz())
        return [0.5 * self._J * ZZ]

    def _collapse_operators(self) -> list[qt.Qobj]:
        """Collect Lindblad collapse operators for both qubits, embedded in the 4D space."""
        return self.q0.collapse_operators(
            embed=self._on_q0
        ) + self.q1.collapse_operators(embed=self._on_q1)

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def evolve(
        self,
        t: npt.NDArray[np.float64],
        wave_q0: npt.NDArray[np.complex128],
        wave_q1: npt.NDArray[np.complex128],
        drive_freq_q0: float,
        drive_freq_q1: float,
        coupling_on: bool = False,
    ) -> None:
        """Propagate the joint state forward under two drive pulses, updating ``self.state``.

        The propagation starts from the current ``self.state``, so successive
        calls chain automatically.  Use `reset` to return to |00>.

        The time grid ``t`` should start at 0 — it is interpreted as relative
        time within the pulse, not absolute lab time.

        Args:
            t:             Time grid [s], shape ``(N,)``, starting at 0.
            wave_q0:       Complex drive envelope for qubit 0, shape ``(N,)``.
            wave_q1:       Complex drive envelope for qubit 1, shape ``(N,)``.
            drive_freq_q0: Drive frequency for qubit 0 [Hz].
            drive_freq_q1: Drive frequency for qubit 1 [Hz].
            coupling_on:   Whether the qubit-qubit ZZ coupling is active during
                           this segment. default ``False`` means single-qubit
                           pulses do not pick up unwanted coupling phase.
        """
        H = self._drive_hamiltonian(wave_q0, wave_q1, drive_freq_q0, drive_freq_q1)
        if coupling_on:
            H += self._coupling_hamiltonian()
        c_ops = self._collapse_operators()
        result = qt.mesolve(H, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    def cphase(
        self, j_coupling: float | None = None, n_steps: int = _WAIT_STEPS
    ) -> None:
        """Apply a CPHASE gate by turning on ZZ coupling for time pi/(2J).

        This represents the flux-pulse-mediated entangling gate used in
        superconducting hardware: the coupling is briefly switched on for
        exactly the right duration to accumulate a pi-phase on |11>.

        Args:
            j_coupling: ZZ coupling strength to use [rad/s].  Defaults to the
                        pair's ``_J``.
            n_steps:    Number of time steps for the mesolve integration.
        """
        J = self._J if j_coupling is None else j_coupling
        duration = np.pi / (2 * J)
        t = np.linspace(0, duration, n_steps)
        H = self._coupling_hamiltonian()
        c_ops = self._collapse_operators()
        result = qt.mesolve(H, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    def wait(
        self, duration: float, n_steps: int = _WAIT_STEPS, coupling_on: bool = False
    ) -> None:
        """Let both qubits decay freely for ``duration`` seconds under zero drive.

        No pulses are applied — only the T1 / T2 Lindblad channels of each qubit
        act.  By default the qubit-qubit coupling is OFF (matching real
        flux-tunable hardware where the coupling is gated).  Set ``coupling_on``
        to True to model an entangling-gate-equivalent free evolution.

        Like `evolve`, this updates ``self.state`` in-place so calls chain.

        Args:
            duration:    Idle time [s].
            n_steps:     Number of time steps for the mesolve integration.
            coupling_on: Whether to include the qubit-qubit ZZ coupling.
        """
        t = np.linspace(0, duration, n_steps)
        H = (
            self._coupling_hamiltonian()
            if coupling_on
            else [qt.tensor(qt.qeye(2), qt.qeye(2)) * 0]
        )
        c_ops = self._collapse_operators()
        result = qt.mesolve(H, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, shots: int = 1) -> npt.NDArray[np.int8]:
        """Sample computational-basis outcomes from the current state.

        Reads ``self.state`` at call time, so the result always reflects the
        most recent `evolve` or `wait`.  Can be called multiple
        times after a single evolution without side effects on the state.

        Each shot yields a pair of bits (b0, b1) ∈ {0, 1}², drawn from the
        Born-rule probabilities of the stored density matrix, then corrupted
        by each qubit's independent readout error model.

        Args:
            shots: Number of two-qubit measurement shots.

        Returns:
            Integer array of shape ``(shots, 2)``; column 0 is qubit 0's outcome,
            column 1 is qubit 1's outcome.
        """
        probs = self._outcome_probabilities()
        outcomes = np.random.choice(4, size=shots, p=probs)

        # Decode the 2-bit outcomes: outcome k → (k >> 1, k & 1)
        bits = np.stack([(outcomes >> 1) & 1, outcomes & 1], axis=1).astype(np.int8)
        bits[:, 0] = self.q0._apply_readout_error(bits[:, 0])
        bits[:, 1] = self.q1._apply_readout_error(bits[:, 1])
        return bits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outcome_probabilities(self) -> npt.NDArray[np.float64]:
        """Compute Born-rule probabilities for all four computational-basis outcomes.

        Returns:
            Normalised probability array ``[P(00), P(01), P(10), P(11)]``.
        """
        projectors = [
            qt.tensor(qt.basis(2, a).proj(), qt.basis(2, b).proj())
            for a in (0, 1)
            for b in (0, 1)
        ]
        probs = np.array([abs((P * self.state).tr()) for P in projectors], dtype=float)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        return probs
