//
//  MindMapView.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SwiftUI
import SpriteKit

/// UIViewRepresentable wrapper for SKView displaying the mind map
/// Provides SwiftUI integration for the SpriteKit visualization
struct MindMapView: View {

    // MARK: - Properties

    @Bindable var viewModel: MindMapViewModel

    // MARK: - Body

    var body: some View {
        GeometryReader { geometry in
            SpriteKitContainer(
                viewModel: viewModel,
                size: geometry.size
            )
        }
        .ignoresSafeArea(edges: .horizontal)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button {
                    NotificationCenter.default.post(name: Notification.Name("MindMap.CenterOnSelection"), object: nil)
                } label: {
                    Label("Center", systemImage: "scope")
                }
                Button {
                    NotificationCenter.default.post(name: Notification.Name("MindMap.ZoomToFit"), object: nil)
                } label: {
                    Label("Fit", systemImage: "dot.viewfinder")
                }
            }
        }
        .onChange(of: viewModel.thoughts.count) { _ in }
        .onChange(of: viewModel.connections.count) { _ in }
    }
}

// MARK: - SpriteKit Container

/// UIViewRepresentable that hosts the SKView and MindMapScene
struct SpriteKitContainer: UIViewRepresentable {

    @Bindable var viewModel: MindMapViewModel
    let size: CGSize

    func makeUIView(context: Context) -> SKView {
        let skView = SKView()
        skView.ignoresSiblingOrder = true
        skView.showsFPS = AppConfig.showDebugInfo
        skView.showsNodeCount = AppConfig.showDebugInfo
        skView.backgroundColor = .clear

        let scene = MindMapScene(size: size)
        scene.scaleMode = .resizeFill
        scene.viewModel = viewModel
        skView.presentScene(scene)
        scene.syncWithViewModel()

        // Store scene reference for updates
        context.coordinator.scene = scene

        NotificationCenter.default.addObserver(forName: Notification.Name("MindMap.CenterOnSelection"), object: nil, queue: .main) { _ in
            context.coordinator.centerOnSelection()
        }
        NotificationCenter.default.addObserver(forName: Notification.Name("MindMap.ZoomToFit"), object: nil, queue: .main) { _ in
            context.coordinator.zoomToFit()
        }

        return skView
    }

    func updateUIView(_ skView: SKView, context: Context) {
        // Update scene size if geometry changed
        if let scene = context.coordinator.scene {
            scene.size = size
            scene.syncWithViewModel()
        }
        context.coordinator.dataDidChange()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        weak var scene: MindMapScene?

        func dataDidChange() {
            scene?.syncWithViewModel()
        }

        func centerOnSelection() {
            scene?.centerOnSelectedNode()
        }

        func zoomToFit() {
            scene?.zoomToFitAllNodes()
        }
    }
}

#Preview {
    MindMapView(viewModel: MindMapViewModel())
}
