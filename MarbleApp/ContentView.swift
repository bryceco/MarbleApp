import RealityKit
import SwiftUI

struct ContentView: View {
	@State private var model = GameModel()

	var body: some View {
		RealityView { content in
			model.makeScene(content)
		} update: { content in
			// Reading tapCount is what makes this closure re-run on each tap.
			_ = model.tapCount
			model.consumePendingTap(content)
		}
		.onTapGesture { location in
			model.tap(at: location)
		}
		.ignoresSafeArea()
		.overlay(alignment: .bottom) {
			controls
		}
	}

	private var controls: some View {
		HStack(spacing: 24) {
			Stepper("Hills: \(model.hillCount)", value: $model.hillCount, in: 0 ... 40)
			Stepper("Height: \(model.hillHeight, specifier: "%.1f")",
			        value: $model.hillHeight, in: 0.2 ... 4.0, step: 0.2)
			Button("New seed") { model.newSeed() }
		}
		.fixedSize()
		.padding()
		.background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
		.padding()
	}
}

#Preview {
	ContentView()
}
