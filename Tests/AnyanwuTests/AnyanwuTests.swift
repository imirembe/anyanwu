import XCTest
@testable import Anyanwu

final class AnyanwuTests: XCTestCase {
    func testServiceAddressRequiresHTTPSExceptLoopback() throws {
        XCTAssertEqual(try AssistantService.validatedURL("https://anyanwu.example/api").host, "anyanwu.example")
        XCTAssertEqual(try AssistantService.validatedURL("http://localhost:8787").port, 8787)
        for invalid in ["", "http://public.example", "file:///private", "https://user:secret@host.example", "https://example.com?key=secret", "https://example.com#fragment"] {
            XCTAssertThrowsError(try AssistantService.validatedURL(invalid), invalid)
        }
    }

    @MainActor func testPreviewIsLabelledAndPersistsAcrossLaunches() async throws {
        let repository = temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository.directory) }
        let store = AssistantStore(repository: repository)
        await store.waitForIdle()
        store.sendText("Teach me an Igbo greeting.")
        await store.waitForIdle()
        XCTAssertEqual(store.messages.count, 2)
        XCTAssertTrue(store.messages[1].isPreview)
        XCTAssertFalse(store.messages[0].isPreview)
        XCTAssertEqual(store.messages[0].text, "Teach me an Igbo greeting.")
        let restored = AssistantStore(repository: repository)
        await restored.waitForIdle()
        XCTAssertEqual(restored.conversations.count, 1)
        restored.open(restored.conversations[0])
        await restored.waitForIdle()
        XCTAssertEqual(restored.messages, store.messages)
    }

    @MainActor func testClearHistoryRemovesRecordingsAndSavedMessages() async throws {
        let repository = temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository.directory) }
        let store = AssistantStore(repository: repository)
        await store.waitForIdle()
        store.sendText("Ndewo")
        await store.waitForIdle()
        try Data([1, 2, 3]).write(to: repository.audioURL("fixture.m4a"))
        store.clearHistory()
        await store.waitForIdle()
        XCTAssertTrue(store.conversations.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: repository.audioURL("fixture.m4a").path))
        let page = try await ConversationArchive(repository: repository).prepare(historyEnabled: true)
        XCTAssertTrue(page.items.isEmpty)
    }

    @MainActor func testHistoryOffDoesNotPersistNewMessages() async throws {
        let repository = temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository.directory) }
        let store = AssistantStore(repository: repository)
        await store.waitForIdle()
        var settings = store.settings
        settings.saveHistory = false
        await store.saveSettings(settings).value
        store.sendText("Ndewo")
        await store.waitForIdle()
        XCTAssertEqual(store.messages.count, 2)
        let restored = AssistantStore(repository: repository)
        await restored.waitForIdle()
        XCTAssertTrue(restored.conversations.isEmpty)
    }

    func testAudioPathsCannotEscapeAudioDirectory() {
        let repository = temporaryRepository()
        XCTAssertEqual(repository.audioURL("../../secrets.m4a").deletingLastPathComponent(), repository.audioDirectory)
    }

    func testOlderSettingsKeepConnectionAndPrivacyPreferences() throws {
        let data = Data(#"{"mode":"Connected","serverURL":"https://voice.example","language":"ig-en","autoplayReplies":false,"saveHistory":false}"#.utf8)
        let settings = try JSONDecoder().decode(AssistantSettings.self, from: data)
        XCTAssertEqual(settings.mode, .connected)
        XCTAssertEqual(settings.serverURL, "https://voice.example")
        XCTAssertFalse(settings.autoplayReplies)
        XCTAssertFalse(settings.saveHistory)
        XCTAssertEqual(settings.interfaceLanguage, .igbo)
    }

    func testBothInterfaceLanguagesRequestBilingualInputAndIgboReplies() throws {
        for language in InterfaceLanguage.allCases {
            let request = TurnRequest(conversationId: "test", interfaceLanguage: language.rawValue,
                                      input: .init(type: "text", text: "Hello, kedu?"), messages: [])
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
            XCTAssertEqual(payload["inputLanguages"] as? [String], ["ig", "en"])
            XCTAssertEqual(payload["responseLanguage"] as? String, "ig")
            XCTAssertEqual(payload["interfaceLanguage"] as? String, language.rawValue)
            XCTAssertNil(payload["language"])
        }
    }

    @MainActor func testEnglishInterfacePersistsWithoutChangingIgboReplies() async throws {
        let repository = temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository.directory) }
        let store = AssistantStore(repository: repository)
        await store.waitForIdle()
        var settings = store.settings
        settings.interfaceLanguage = .english
        await store.saveSettings(settings).value
        store.sendText("Hello")
        await store.waitForIdle()
        XCTAssertEqual(store.settings.interfaceLanguage.greeting, "Hello.")
        XCTAssertEqual(store.messages.last?.text, "Ndewo! Kedu ka ị mere?")
        let restored = AssistantStore(repository: repository)
        await restored.waitForIdle()
        XCTAssertEqual(restored.settings.interfaceLanguage, .english)
        XCTAssertEqual(restored.messages.count, 0)
        restored.open(try XCTUnwrap(restored.conversations.first))
        await restored.waitForIdle()
        XCTAssertEqual(restored.messages.last?.text, "Ndewo! Kedu ka ị mere?")
    }

    private func temporaryRepository() -> LocalRepository {
        LocalRepository(directory: FileManager.default.temporaryDirectory.appending(path: "anyanwu-tests-\(UUID().uuidString)", directoryHint: .isDirectory))
    }
}
