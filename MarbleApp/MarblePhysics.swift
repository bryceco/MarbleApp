import simd

/// Marble state in surface coordinates. Position and velocity live in the
/// x/z plane; height is implied by the surface.
struct MarbleState {
	var p: SIMD2<Float>
	var v: SIMD2<Float>
	/// Total mechanical energy per unit mass, fixed at drop time.
	var energy: Float
}

/// Frictionless point mass constrained to y = h(x, z) under gravity.
///
/// From the Lagrangian L = ½|ṙ|² − g·h with ṙ = (ẋ, ∇h·q̇, ż):
///   q̈ = −∇h · (g + q̇ᵀ H q̇) / (1 + |∇h|²)
/// Integrated with RK4, then the speed is rescaled so the total energy is
/// exactly the drop-time value, so there's no drift and the marble never stops.
enum MarblePhysics {
	static let gravity: Float = 9.8

	static func acceleration(p: SIMD2<Float>, v: SIMD2<Float>, in field: Heightfield) -> SIMD2<Float> {
		let s = field.sample(at: p)
		let curvature = dot(v, s.hess * v)
		return -s.grad * (gravity + curvature) / (1 + dot(s.grad, s.grad))
	}

	/// Kinetic energy per unit mass: ½ |(ẋ, ∇h·q̇, ż)|²
	static func kineticEnergy(v: SIMD2<Float>, grad: SIMD2<Float>) -> Float {
		let vy = dot(grad, v)
		return 0.5 * (dot(v, v) + vy * vy)
	}

	static func step(_ m: inout MarbleState, dt: Float, in field: Heightfield) {
		let k1v = acceleration(p: m.p, v: m.v, in: field)
		let k1p = m.v

		let k2v = acceleration(p: m.p + 0.5 * dt * k1p, v: m.v + 0.5 * dt * k1v, in: field)
		let k2p = m.v + 0.5 * dt * k1v

		let k3v = acceleration(p: m.p + 0.5 * dt * k2p, v: m.v + 0.5 * dt * k2v, in: field)
		let k3p = m.v + 0.5 * dt * k2v

		let k4v = acceleration(p: m.p + dt * k3p, v: m.v + dt * k3v, in: field)
		let k4p = m.v + dt * k3v

		m.p += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
		m.v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

		// Safety net; the rim should keep the marble inside on its own.
		m.p = clamp(m.p, min: -field.halfExtent, max: field.halfExtent)

		// Project back onto the energy shell.
		let s = field.sample(at: m.p)
		let ke = kineticEnergy(v: m.v, grad: s.grad)
		let target = m.energy - gravity * s.h
		if target > 0, ke > 1e-9 {
			m.v *= sqrt(target / ke)
		}
	}
}
