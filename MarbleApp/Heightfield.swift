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
/// a rim of constant height along all four edges that is flat in the interior,
/// plus a sum of Gaussian bumps (positive = hill, negative = valley).
struct Heightfield {
	struct Bump {
		var center: SIMD2<Float>
		var amplitude: Float
		var sigma: Float
	}

	/// The playfield spans [-halfExtent, +halfExtent] in x and z.
	let halfExtent: SIMD2<Float>
	/// Height of the rim along the edges, a little above the tallest hill.
	let rimHeight: Float
	let bumps: [Bump]

	init(halfExtent: SIMD2<Float>, hillCount: Int, hillHeight: Float, seed: UInt64) {
		self.halfExtent = halfExtent
		rimHeight = hillHeight * 1.25
		var rng = SplitMix64(seed: seed)
		let minExtent = min(halfExtent.x, halfExtent.y)
		bumps = (0 ..< hillCount).map { _ in
			let cx = Float.random(in: -0.75 ... 0.75, using: &rng) * halfExtent.x
			let cz = Float.random(in: -0.75 ... 0.75, using: &rng) * halfExtent.y
			let sign: Float = Bool.random(using: &rng) ? 1 : -1
			let amplitude = sign * hillHeight * Float.random(in: 0.5 ... 1.0, using: &rng)
			let sigma = Float.random(in: 0.12 ... 0.25, using: &rng) * minExtent
			return Bump(center: SIMD2(cx, cz), amplitude: amplitude, sigma: sigma)
		}
	}

	/// Height only; cheaper than `sample` when derivatives aren't needed.
	func height(at p: SIMD2<Float>) -> Float {
		let u = p / halfExtent
		let u8 = Self.pow8(u)
		var h = rimHeight * (1 - (1 - u8.x) * (1 - u8.y))
		for b in bumps {
			let d = p - b.center
			h += b.amplitude * exp(-dot(d, d) / (2 * b.sigma * b.sigma))
		}
		return h
	}

	/// Height, gradient and Hessian at a point.
	func sample(at p: SIMD2<Float>) -> HeightSample {
		// Rim: R·(1 − A·B) with A = 1 − u⁸, B = 1 − v⁸, u = x/hx, v = z/hz.
		// Equals R along every edge, ≈0 in the interior.
		let u = p / halfExtent
		let u2 = u * u
		let u6 = u2 * u2 * u2
		let u7 = u6 * u
		let u8 = u6 * u2
		let a = 1 - u8.x
		let b = 1 - u8.y
		let du = 8 * u7 / halfExtent // d(u⁸)/dx, d(v⁸)/dz
		let ddu = 56 * u6 / (halfExtent * halfExtent) // second derivatives
		var h = rimHeight * (1 - a * b)
		var g = rimHeight * SIMD2(du.x * b, a * du.y)
		var hxx = rimHeight * ddu.x * b
		var hzz = rimHeight * a * ddu.y
		var hxz = -rimHeight * du.x * du.y

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

	private static func pow8(_ u: SIMD2<Float>) -> SIMD2<Float> {
		let u2 = u * u
		let u4 = u2 * u2
		return u4 * u4
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
