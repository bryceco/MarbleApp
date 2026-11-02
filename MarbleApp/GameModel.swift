import CoreGraphics
import Observation
import RealityKit
import SwiftUI

@MainActor
@Observable
final class GameModel {
	let field = Heightfield(halfExtent: SIMD2(8, 6), rimHeight: 4, bumpCount: 10, seed: 2)
	let marbleRadius: Float = 0.3

	/// Bumped by the view on each tap so the RealityView update closure re-runs.
	private(set) var tapCount = 0
	@ObservationIgnored private var pendingTap: CGPoint?

	@ObservationIgnored private var marble: MarbleState?
	@ObservationIgnored private let marbleEntity = ModelEntity()
	@ObservationIgnored private var updateSubscription: EventSubscription?

	// MARK: Scene setup

	func makeScene(_ content: RealityViewCameraContent) {
		// Terrain
		var ground = PhysicallyBasedMaterial()
		ground.baseColor = .init(texture: .init(makeCheckerTexture(
			squares: 32,
			colorA: CGColor(red: 0.55, green: 0.70, blue: 0.45, alpha: 1),
			colorB: CGColor(red: 0.48, green: 0.63, blue: 0.40, alpha: 1))))
		ground.roughness = 0.9
		let terrain = ModelEntity(mesh: field.makeMesh(resolution: 160), materials: [ground])
		content.add(terrain)

		// Marble (hidden until the first tap)
		var shiny = PhysicallyBasedMaterial()
		shiny.baseColor = .init(texture: .init(makeCheckerTexture(
			squares: 8,
			colorA: CGColor(red: 0.85, green: 0.15, blue: 0.15, alpha: 1),
			colorB: CGColor(red: 0.95, green: 0.90, blue: 0.80, alpha: 1))))
		shiny.roughness = 0.25
		shiny.metallic = 0.1
		marbleEntity.model = ModelComponent(mesh: .generateSphere(radius: marbleRadius), materials: [shiny])
		marbleEntity.isEnabled = false
		content.add(marbleEntity)

		// Camera
		let camera = PerspectiveCamera()
		camera.look(at: .zero, from: SIMD3(0, 16, 14), relativeTo: nil)
		content.add(camera)

		// Key light with shadows, plus a dim fill from the other side.
		let key = DirectionalLight()
		key.light.intensity = 3000
		key.shadow = DirectionalLightComponent.Shadow()
		key.look(at: .zero, from: SIMD3(6, 12, 8), relativeTo: nil)
		content.add(key)

		let fill = DirectionalLight()
		fill.light.intensity = 800
		fill.look(at: .zero, from: SIMD3(-8, 6, -4), relativeTo: nil)
		content.add(fill)

		updateSubscription = content.subscribe(to: SceneEvents.Update.self) { [weak self] event in
			self?.tick(deltaTime: event.deltaTime)
		}
	}

	// MARK: Input

	func tap(at point: CGPoint) {
		pendingTap = point
		tapCount += 1
	}

	/// Called from the RealityView update closure, where we have the content
	/// (and therefore the camera) needed to unproject the tap.
	func consumePendingTap(_ content: RealityViewCameraContent) {
		guard let point = pendingTap else { return }
		pendingTap = nil
		guard let ray = content.ray(through: point, in: .local, to: .scene),
		      let p = surfacePoint(along: ray)
		else { return }
		drop(at: p)
	}

	private func drop(at p: SIMD2<Float>) {
		let h = field.height(at: p)
		marble = MarbleState(p: p, v: .zero, energy: MarblePhysics.gravity * h)
		marbleEntity.isEnabled = true
		place(MarbleState(p: p, v: .zero, energy: 0), deltaTime: 0)
	}

	/// March along the ray until it goes below the surface, then bisect.
	private func surfacePoint(along ray: (origin: SIMD3<Float>, direction: SIMD3<Float>)) -> SIMD2<Float>? {
		let dir = normalize(ray.direction)
		func below(_ t: Float) -> Bool? {
			let q = ray.origin + t * dir
			let xz = SIMD2(q.x, q.z)
			guard all(abs(xz) .<= field.halfExtent) else { return nil }
			return q.y < field.height(at: xz)
		}
		var t: Float = 0
		let step: Float = 0.05
		var tPrev: Float?
		while t < 100 {
			if let hit = below(t) {
				if hit, let lo = tPrev {
					var lo = lo, hi = t
					for _ in 0 ..< 20 {
						let mid = (lo + hi) / 2
						if below(mid) == true { hi = mid } else { lo = mid }
					}
					let q = ray.origin + hi * dir
					return SIMD2(q.x, q.z)
				}
				tPrev = t
			} else {
				tPrev = nil
			}
			t += step
		}
		return nil
	}

	// MARK: Simulation

	private func tick(deltaTime: TimeInterval) {
		guard var m = marble else { return }
		let dt = Float(min(deltaTime, 1.0 / 30))
		let substeps = max(1, Int((dt * 240).rounded(.up)))
		let h = dt / Float(substeps)
		for _ in 0 ..< substeps {
			MarblePhysics.step(&m, dt: h, in: field)
		}
		marble = m
		place(m, deltaTime: dt)
	}

	/// Position the marble entity on the surface and roll it without slipping.
	private func place(_ m: MarbleState, deltaTime dt: Float) {
		let s = field.sample(at: m.p)
		let n = Heightfield.normal(forGradient: s.grad)
		marbleEntity.position = SIMD3(m.p.x, s.h, m.p.y) + marbleRadius * n

		let v3 = SIMD3(m.v.x, dot(s.grad, m.v), m.v.y)
		let omega = cross(n, v3) / marbleRadius
		let angle = length(omega) * dt
		if angle > 0 {
			marbleEntity.orientation = simd_quatf(angle: angle, axis: omega / length(omega)) * marbleEntity.orientation
		}
	}
}
