import simd

/// Height and derivatives of the surface at a point.
/// All derivatives are analytic, so the physics never sees a discontinuity.
struct HeightSample {
	/// Surface height (y) at the point.
	var h: Float
	/// (∂h/∂x, ∂h/∂z)
	var grad: SIMD2<Float>
	/// Second derivatives, [[hxx, hxz], [hxz, hzz]]
	var hess: simd_float2x2
}

/// A smooth (C∞) surface over a rectangular playfield in the x/z plane:
/// a quartic bowl that rises toward the edges, plus a sum of Gaussian bumps
/// (positive = hill, negative = valley) in the interior.
struct Heightfield {
	struct Bump {
		var center: SIMD2<Float>
		var amplitude: Float
		var sigma: Float
	}

	/// The playfield spans [-halfExtent, +halfExtent] in x and z.
	let halfExtent: SIMD2<Float>
	/// Height of the bowl term at the edge midpoints (corners are 2x this).
	let rimHeight: Float
	let bumps: [Bump]

	init(halfExtent: SIMD2<Float>, rimHeight: Float, bumpCount: Int, seed: UInt64) {
		self.halfExtent = halfExtent
		self.rimHeight = rimHeight
		var rng = SplitMix64(seed: seed)
		let minExtent = min(halfExtent.x, halfExtent.y)
		bumps = (0 ..< bumpCount).map { _ in
			let cx = Float.random(in: -0.75 ... 0.75, using: &rng) * halfExtent.x
			let cz = Float.random(in: -0.75 ... 0.75, using: &rng) * halfExtent.y
			let sign: Float = Bool.random(using: &rng) ? 1 : -1
			let amplitude = sign * Float.random(in: 0.4 ... 1.2, using: &rng)
			let sigma = Float.random(in: 0.12 ... 0.25, using: &rng) * minExtent
			return Bump(center: SIMD2(cx, cz), amplitude: amplitude, sigma: sigma)
		}
	}

	/// Height only; cheaper than `sample` when derivatives aren't needed.
	func height(at p: SIMD2<Float>) -> Float {
		let u = p / halfExtent
		let u2 = u * u
		var h = rimHeight * (u2.x * u2.x + u2.y * u2.y)
		for b in bumps {
			let d = p - b.center
			h += b.amplitude * exp(-dot(d, d) / (2 * b.sigma * b.sigma))
		}
		return h
	}

	/// Height, gradient and Hessian at a point.
	func sample(at p: SIMD2<Float>) -> HeightSample {
		// Bowl: rimHeight * (u⁴ + v⁴), u = x/hx, v = z/hz
		let u = p / halfExtent
		let u2 = u * u
		let u3 = u2 * u
		var h = rimHeight * (u2.x * u2.x + u2.y * u2.y)
		var g = 4 * rimHeight * u3 / halfExtent
		var hxx = 12 * rimHeight * u2.x / (halfExtent.x * halfExtent.x)
		var hzz = 12 * rimHeight * u2.y / (halfExtent.y * halfExtent.y)
		var hxz: Float = 0

		// Gaussian bumps: e = a·exp(-|d|²/2σ²)
		//   ∇e = -e·d/σ²,  H(e) = e·(d dᵀ/σ⁴ − I/σ²)
		for b in bumps {
			let d = p - b.center
			let s2 = b.sigma * b.sigma
			let s4 = s2 * s2
			let e = b.amplitude * exp(-dot(d, d) / (2 * s2))
			h += e
			g -= e * d / s2
			hxx += e * (d.x * d.x / s4 - 1 / s2)
			hzz += e * (d.y * d.y / s4 - 1 / s2)
			hxz += e * (d.x * d.y / s4)
		}
		return HeightSample(h: h,
		                    grad: g,
		                    hess: simd_float2x2(rows: [SIMD2(hxx, hxz), SIMD2(hxz, hzz)]))
	}

	/// Unit surface normal for a gradient.
	static func normal(forGradient g: SIMD2<Float>) -> SIMD3<Float> {
		normalize(SIMD3(-g.x, 1, -g.y))
	}
}

/// Small seeded generator so a given seed always produces the same surface.
struct SplitMix64: RandomNumberGenerator {
	private var state: UInt64

	init(seed: UInt64) {
		state = seed
	}

	mutating func next() -> UInt64 {
		state &+= 0x9E37_79B9_7F4A_7C15
		var z = state
		z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
		z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
		return z ^ (z >> 31)
	}
}
