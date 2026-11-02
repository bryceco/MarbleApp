import CoreGraphics
import RealityKit

extension Heightfield {
	/// Tessellates the surface into a RealityKit mesh with analytic normals.
	func makeMesh(resolution: Int) -> MeshResource {
		let n = resolution
		var positions: [SIMD3<Float>] = []
		var normals: [SIMD3<Float>] = []
		var uvs: [SIMD2<Float>] = []
		positions.reserveCapacity((n + 1) * (n + 1))
		normals.reserveCapacity(positions.capacity)
		uvs.reserveCapacity(positions.capacity)

		for j in 0 ... n {
			for i in 0 ... n {
				let t = SIMD2(Float(i), Float(j)) / Float(n)
				let p = (t * 2 - 1) * halfExtent
				let s = sample(at: p)
				positions.append(SIMD3(p.x, s.h, p.y))
				normals.append(Heightfield.normal(forGradient: s.grad))
				uvs.append(t)
			}
		}

		// Two counter-clockwise (viewed from +y) triangles per quad.
		var indices: [UInt32] = []
		indices.reserveCapacity(n * n * 6)
		for j in 0 ..< n {
			for i in 0 ..< n {
				let a = UInt32(j * (n + 1) + i)
				let b = a + 1
				let c = a + UInt32(n + 1)
				let d = c + 1
				indices += [a, c, b, b, c, d]
			}
		}

		var desc = MeshDescriptor(name: "terrain")
		desc.positions = MeshBuffer(positions)
		desc.normals = MeshBuffer(normals)
		desc.textureCoordinates = MeshBuffer(uvs)
		desc.primitives = .triangles(indices)
		return try! MeshResource.generate(from: [desc])
	}
}

/// A checkerboard texture, used to make rolling and surface shape visible.
func makeCheckerTexture(squares: Int, size: Int = 256,
                        colorA: CGColor, colorB: CGColor) -> TextureResource
{
	let ctx = CGContext(data: nil, width: size, height: size,
	                    bitsPerComponent: 8, bytesPerRow: 0,
	                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
	                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
	let cell = CGFloat(size) / CGFloat(squares)
	for j in 0 ..< squares {
		for i in 0 ..< squares {
			ctx.setFillColor((i + j) % 2 == 0 ? colorA : colorB)
			ctx.fill(CGRect(x: CGFloat(i) * cell, y: CGFloat(j) * cell, width: cell, height: cell))
		}
	}
	return try! TextureResource(image: ctx.makeImage()!, options: .init(semantic: .color))
}
