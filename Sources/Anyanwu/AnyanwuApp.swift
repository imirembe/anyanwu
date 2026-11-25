import AppIntents
import SwiftUI

@main
struct AnyanwuApp: App {
    @State private var assistant = AssistantStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(assistant)
                .preferredColorScheme(.light)
                .onChange(of: scenePhase) { _, phase in
                    switch phase {
                    case .background: assistant.suspendForBackground()
                    case .inactive:
                        if assistant.recorder.isRecording { assistant.stopRecordingForInterruption() }
                    case .active: assistant.resumeForeground()
                    @unknown default: break
                    }
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                    assistant.handleMemoryPressure()
                }
                .task { AnyanwuShortcuts.updateAppShortcutParameters() }
        }
    }
}
