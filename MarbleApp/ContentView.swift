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
	}
}

#Preview {
	ContentView()
}
