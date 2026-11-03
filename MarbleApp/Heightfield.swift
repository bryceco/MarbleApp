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
		var h = rim(p / halfExtent).h
		for b in bumps {
			let d = p - b.center
			h += b.amplitude * exp(-dot(d, d) / (2 * b.sigma * b.sigma))
		}
		return h
	}

	/// Height, gradient and Hessian at a point.
	func sample(at p: SIMD2<Float>) -> HeightSample {
		let r = rim(p / halfExtent)
		var h = r.h
		var g = r.grad
		var hxx = r.hess[0, 0]
		var hzz = r.hess[1, 1]
		var hxz = r.hess[0, 1]

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

	/// Rim term in normalized coordinates u = x/hx, v = z/hz:
	///   h = R · s^(1/8),  s = u³² + v³²
	/// i.e. R times a smooth approximation of max(|u|, |v|)⁴. It is ≈0 in the
	/// interior, R along the edges, 2^(1/8) ≈ 1.09·R at the corners, and keeps
	/// rising past the edge everywhere, so there is always an inward force.
	private func rim(_ u: SIMD2<Float>) -> HeightSample {
		let u2 = u * u
		let u4 = u2 * u2
		let u8 = u4 * u4
		let u16 = u8 * u8
		let u30 = u16 * u8 * u4 * u2
		let u31 = u30 * u
		let u32 = u30 * u2
		let s = u32.x + u32.y
		// Below this the rim is < 1e-3·R; skip to avoid 0·inf in the derivatives.
		guard s > 1e-24 else { return HeightSample(h: 0, grad: .zero, hess: .init()) }

		// With h = R·s^q: ∇h = h·q·(∇s/s), and the Hessian follows by the
		// product rule. Working with ∇s/s keeps everything bounded near s → 0.
		let q: Float = 1.0 / 8
		let h = rimHeight * s.squareRoot().squareRoot().squareRoot()
		let a = 32 * u31 / (s * halfExtent) // (∂s/∂x)/s, (∂s/∂z)/s
		let b = 32 * 31 * u30 / (s * halfExtent * halfExtent) // (∂²s/∂x²)/s, (∂²s/∂z²)/s
		let hq = h * q
		let grad = hq * a
		let hxx = hq * ((q - 1) * a.x * a.x + b.x)
		let hzz = hq * ((q - 1) * a.y * a.y + b.y)
		let hxz = hq * (q - 1) * a.x * a.y
		return HeightSample(h: h, grad: grad,
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
