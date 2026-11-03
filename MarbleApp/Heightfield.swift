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

	static let zero = HeightSample(h: 0, grad: .zero, hess: .init())

	static func + (a: Self, b: Self) -> Self {
		HeightSample(h: a.h + b.h, grad: a.grad + b.grad, hess: a.hess + b.hess)
	}

	static func * (k: Float, a: Self) -> Self {
		HeightSample(h: k * a.h, grad: k * a.grad, hess: k * a.hess)
	}

	/// Product rule for two scalar fields.
	static func * (a: Self, b: Self) -> Self {
		let cross = simd_float2x2(columns: (a.grad.x * b.grad, a.grad.y * b.grad))
		return HeightSample(h: a.h * b.h,
		                    grad: a.h * b.grad + b.h * a.grad,
		                    hess: a.h * b.hess + b.h * a.hess + cross + cross.transpose)
	}
}

/// A smooth (C∞) surface over a rectangular playfield in the x/z plane:
///   h = R · rim(p) + window(p) · bumps(p)
/// The rim is ≈0 in the interior and R along every edge; the window fades the
/// Gaussian bumps (positive = hill, negative = valley) to zero at the edge; and
/// R is chosen so the edge is the highest point of the surface, which with
/// energy conservation guarantees the marble can never reach it.
struct Heightfield {
	struct Bump {
		var center: SIMD2<Float>
		var amplitude: Float
		var sigma: Float
	}

	/// The playfield spans [-halfExtent, +halfExtent] in x and z.
	let halfExtent: SIMD2<Float>
	/// Height of the rim along the edges.
	let rimHeight: Float
	let bumps: [Bump]

	init(halfExtent: SIMD2<Float>, hillCount: Int, hillHeight: Float, seed: UInt64) {
		self.halfExtent = halfExtent
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

		// With h = R·r + w·b and r < 1 inside, h ≤ R everywhere iff
		// R ≥ w·b / (1 − r) everywhere. Scan a grid for the worst case.
		var required: Float = 0
		let n = 96
		for j in 0 ... n {
			for i in 0 ... n {
				let u = SIMD2(Float(i), Float(j)) / Float(n) * 1.98 - 0.99
				let r = Self.unitRim(u).h
				let wb = Self.window(u).h * Self.bumpSum(bumps, at: u * halfExtent).h
				if wb > 0, r < 0.999 {
					required = max(required, wb / (1 - r))
				}
			}
		}
		rimHeight = max(required * 1.05, hillHeight * 0.5)
	}

	/// Height only.
	func height(at p: SIMD2<Float>) -> Float {
		sample(at: p).h
	}

	/// Height, gradient and Hessian at a point.
	func sample(at p: SIMD2<Float>) -> HeightSample {
		let u = p / halfExtent
		var rim = rimHeight * Self.unitRim(u)
		rim.grad /= halfExtent
		rim.hess = scaleHessian(rim.hess)
		var win = Self.window(u)
		win.grad /= halfExtent
		win.hess = scaleHessian(win.hess)
		return rim + win * Self.bumpSum(bumps, at: p)
	}

	/// Unit surface normal for a gradient.
	static func normal(forGradient g: SIMD2<Float>) -> SIMD3<Float> {
		normalize(SIMD3(-g.x, 1, -g.y))
	}

	// MARK: Terms

	/// Sum of Gaussian bumps in world units:
	///   e = a·exp(-|d|²/2σ²),  ∇e = -e·d/σ²,  H(e) = e·(d dᵀ/σ⁴ − I/σ²)
	private static func bumpSum(_ bumps: [Bump], at p: SIMD2<Float>) -> HeightSample {
		var h: Float = 0
		var g = SIMD2<Float>.zero
		var hxx: Float = 0, hzz: Float = 0, hxz: Float = 0
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
		return HeightSample(h: h, grad: g, hess: simd_float2x2(rows: [SIMD2(hxx, hxz), SIMD2(hxz, hzz)]))
	}

	/// Rim shape in normalized coordinates u = x/hx, v = z/hz, derivatives w.r.t. u, v:
	///   r = s^(1/8),  s = u³² + v³²
	/// a smooth approximation of max(|u|, |v|)⁴: ≈0 in the interior, 1 along the
	/// edges, 2^(1/8) ≈ 1.09 at the corners, and still rising past the edge.
	private static func unitRim(_ u: SIMD2<Float>) -> HeightSample {
		let u2 = u * u
		let u4 = u2 * u2
		let u8 = u4 * u4
		let u16 = u8 * u8
		let u30 = u16 * u8 * u4 * u2
		let u31 = u30 * u
		let u32 = u30 * u2
		let s = u32.x + u32.y
		// Below this the rim is < 1e-3; skip to avoid 0·inf in the derivatives.
		guard s > 1e-24 else { return .zero }

		// With r = s^q: ∇r = r·q·(∇s/s), and the Hessian follows by the
		// product rule. Working with ∇s/s keeps everything bounded near s → 0.
		let q: Float = 1.0 / 8
		let r = s.squareRoot().squareRoot().squareRoot()
		let a = 32 * u31 / s
		let b = 32 * 31 * u30 / s
		let rq = r * q
		let hxx = rq * ((q - 1) * a.x * a.x + b.x)
		let hzz = rq * ((q - 1) * a.y * a.y + b.y)
		let hxz = rq * (q - 1) * a.x * a.y
		return HeightSample(h: r, grad: rq * a, hess: simd_float2x2(rows: [SIMD2(hxx, hxz), SIMD2(hxz, hzz)]))
	}

	/// Bump window in normalized coordinates, derivatives w.r.t. u, v:
	///   w = (1 − u⁸)(1 − v⁸)
	/// 1 in the interior, 0 along the edges.
	private static func window(_ u: SIMD2<Float>) -> HeightSample {
		let u2 = u * u
		let u6 = u2 * u2 * u2
		let u7 = u6 * u
		let u8 = u6 * u2
		let a = 1 - u8.x
		let b = 1 - u8.y
		let da = -8 * u7 // d(1−u⁸)/du
		let dda = -56 * u6
		return HeightSample(h: a * b,
		                    grad: SIMD2(da.x * b, a * da.y),
		                    hess: simd_float2x2(rows: [SIMD2(dda.x * b, da.x * da.y),
		                                              SIMD2(da.x * da.y, a * dda.y)]))
	}

	/// Convert a Hessian taken w.r.t. (u, v) to one w.r.t. (x, z).
	private func scaleHessian(_ m: simd_float2x2) -> simd_float2x2 {
		let s = simd_float2x2(diagonal: 1 / halfExtent)
		return s * m * s
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
