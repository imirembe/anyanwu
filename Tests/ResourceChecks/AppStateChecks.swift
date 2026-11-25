import Foundation

@MainActor enum AppStateChecks {
    static func run(_ directory: URL) async throws {
        let repository = LocalRepository(directory: directory)
        let store = AssistantStore(repository: repository)
        defer { store.suspendForBackground() }
        await store.waitForIdle()
        try check(!store.isPreparingStorage && store.error == nil, "store initialization")
        var preview = store.settings
        preview.mode = .preview
        await store.saveSettings(preview).value
        let modelSentinel = repository.directory.appending(path: "OnDeviceSpeech/active/model-fixture")
        try FileManager.default.createDirectory(at: modelSentinel.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([1]).write(to: modelSentinel)
        for _ in 0..<30 { store.sendText("Hello"); await store.waitForIdle() }
        try check(store.messages.count == 50 && store.hasOlderMessages, "store retained more than a message page")
        try check(store.conversations.first?.messageCount == 60, "store dropped saved messages")
        store.showOlderMessages()
        await store.waitForIdle()
        try check(store.messages.count == 10 && store.hasNewerMessages && !store.hasOlderMessages, "store older window")
        store.sendText("Thank you")
        await store.waitForIdle()
        try check(store.messages.count == 50 && !store.hasNewerMessages, "sending from older page must return to latest")
        try check(store.conversations.first?.messageCount == 62 && store.messages.last?.text == "Daalụ!", "saved reply count")
        var preferences = store.settings
        preferences.interfaceLanguage = .english
        await store.saveSettings(preferences).value
        let restored = AssistantStore(repository: repository)
        defer { restored.suspendForBackground() }
        await restored.waitForIdle()
        try check(restored.settings.interfaceLanguage == .english, "interface setting did not persist")
        restored.open(restored.conversations[0])
        await restored.waitForIdle()
        try check(restored.messages.count == 50 && restored.messages.last?.text == "Daalụ!", "restored history page")
        preferences.saveHistory = false
        await store.saveSettings(preferences).value
        try check(store.messages.isEmpty && store.conversations.isEmpty, "history opt-out did not clear state")
        for _ in 0..<100 { store.sendText("Hello"); await store.waitForIdle() }
        try check(store.messages.count == 200 && store.conversations.isEmpty, "history-off session limit")
        store.sendText("Over limit")
        await store.waitForIdle()
        try check(store.messages.count == 200 && store.error != nil, "session grew past its cap")
        store.newConversation()
        store.sendText("Ndewo")
        await store.waitForIdle()
        preferences.saveHistory = true
        await store.saveSettings(preferences).value
        try check(store.conversations.first?.messageCount == 2, "opt-in did not preserve current session")
        store.clearHistory()
        await store.waitForIdle()
        try check(store.messages.isEmpty && store.conversations.isEmpty, "clear history did not finish")
        try check(FileManager.default.fileExists(atPath: modelSentinel.path), "history cleanup erased a downloaded model")
        store.sendText("Hello")
        await store.waitForIdle()
        try check(store.conversations.first?.messageCount == 2, "archive unusable after clear")
        store.toggleRecording()
        try check(store.showConversation && store.isStartingRecording && !store.isTextEntryVisible,
                  "voice capture did not immediately open the conversation")
        await store.waitForIdle()
        try check(store.recorder.isRecording, "recording did not start")
        guard let recordingID = store.recordingSessionID else { throw CheckFailure(description: "recording session missing") }
        store.receivePartialTranscript("Ndewo, kedu?", sessionID: recordingID)
        try check(store.recordingTranscript == "Ndewo, kedu?", "partial transcript not presented")
        store.receivePartialTranscript("stale result", sessionID: UUID())
        try check(store.recordingTranscript == "Ndewo, kedu?", "stale partial transcript overwrote the current recording")
        try check(!store.messages.contains(where: { $0.text == "Ndewo, kedu?" }), "provisional transcript became history")
        store.finishRecording()
        await store.waitForIdle()
        try check(store.recordingTranscript.isEmpty && store.recordingSessionID == nil && store.transcribingMessageID == nil,
                  "completed voice turn retained provisional transcription state")
        await store.refreshStorageUsage()
        try check(store.storageUsage.temporaryAudioBytes > 0, "preview input clip missing")
        store.navigate(to: .settings)
        await store.waitForIdle()
        try check(store.storageUsage.temporaryAudioBytes == 0 && !store.isWorking, "navigation left temporary clips or work behind")
        store.receivePartialTranscript("late result", sessionID: recordingID)
        try check(store.recordingTranscript.isEmpty, "transcript callback survived navigation")
        store.newConversation()
        store.toggleRecording()
        store.cancelRecording()
        await store.waitForIdle()
        try check(!store.isStartingRecording && !store.recorder.isRecording && store.recordingSessionID == nil,
                  "canceled microphone start continued recording")
        store.cancelSpeechPreparation()
        await store.waitForIdle()
        try check(!store.settings.automaticallyPrepareSpeech && !(try repository.loadSettings()).automaticallyPrepareSpeech,
                  "canceling setup did not persist download opt-out")
        store.prepareSpeech()
        await store.waitForIdle()
        try check((try repository.loadSettings()).automaticallyPrepareSpeech, "explicit preparation did not re-enable setup")
        store.removeOnDeviceSpeech()
        await store.waitForIdle()
        try check(!(try repository.loadSettings()).automaticallyPrepareSpeech, "model removal did not persist download opt-out")
        var apple = store.settings; apple.mode = .apple
        await store.saveSettings(apple).value
        store.error = nil
        store.toggleRecording()
        await store.waitForIdle()
        try check(store.showConversation && !store.isStartingRecording && !store.recorder.isRecording && store.error == nil,
                  "missing speech model must present setup without starting a microphone or an error alert")
    }
}
