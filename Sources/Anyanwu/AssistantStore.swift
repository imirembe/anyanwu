import Foundation
import Observation

@MainActor @Observable
final class AssistantStore {
    var tab: AppTab = .speak
    var conversations: [ConversationSummary] = []
    var messages: [Message] = []
    var activeConversationID: UUID?
    var settings: AssistantSettings
    var draft = ""
    var isWorking = false
    var isPreparingStorage = true
    var isLoadingConversation = false
    var isLoadingHistory = false
    var hasOlderMessages = false
    var hasNewerMessages = false
    var hasOlderConversations = false
    var hasNewerConversations = false
    var storageUsage = StorageUsage()
    var error: String?
    var recordingNotice: String?
    var showConversation = false
    var isTextEntryVisible = false
    var isStartingRecording = false
    private(set) var recordingTranscript = ""
    private(set) var transcribingMessageID: UUID?
    var recordingSessionID: UUID? { isStartingRecording || recorder.isRecording ? recordingGeneration : nil }
    let recorder = VoiceRecorder()
    let playback = VoicePlayback()
    let appleAssistant: AppleAssistant
    let repository: LocalRepository
    private let archive: ConversationArchive
    private let service = AssistantService()
    @ObservationIgnored private var startupTask: Task<Void, Never>?
    @ObservationIgnored private var settingsTask: Task<Void, Never>?
    @ObservationIgnored private var pendingTurn: Task<Void, Never>?
    @ObservationIgnored private var pageTask: Task<Void, Never>?
    @ObservationIgnored private var recordingTask: Task<Void, Never>?
    @ObservationIgnored private var maintenanceTask: Task<Void, Never>?
    @ObservationIgnored private var turnGeneration = UUID()
    @ObservationIgnored private var pageGeneration = UUID()
    @ObservationIgnored private var recordingGeneration = UUID()
    @ObservationIgnored private var activeCreatedAt = Date.now
    @ObservationIgnored private var recordingURL: URL?
    @ObservationIgnored private var inputFilename: String?
    @ObservationIgnored private var storageAvailable = false
    @ObservationIgnored private var isForeground = true
    private var language: InterfaceLanguage { settings.interfaceLanguage }
    private var retainsHistory: Bool { settings.saveHistory && storageAvailable }

    init(repository: LocalRepository = .init()) {
        let arguments = ProcessInfo.processInfo.arguments
        let testing = arguments.contains("-anyanwu-ui-testing")
        let storage = testing ? LocalRepository(directory: FileManager.default.temporaryDirectory.appending(path: "anyanwu-ui-\(UUID().uuidString)", directoryHint: .isDirectory)) : repository
        self.repository = storage
        appleAssistant = AppleAssistant(directory: storage.directory)
        archive = ConversationArchive(repository: storage)
        var initial = AssistantSettings()
        var initialError: String?
        do { if !testing { initial = try storage.loadSettings() } }
        catch { initialError = "Settings could not be read. Existing files have been preserved." }
        if testing, let index = arguments.firstIndex(of: "-anyanwu-ui-language"), arguments.indices.contains(index + 1),
           let selected = InterfaceLanguage(rawValue: arguments[index + 1]) { initial.interfaceLanguage = selected }
        settings = initial
        if testing { settings.mode = .preview }
        #if DEBUG
        if testing && arguments.contains("-anyanwu-ui-apple-settings") {
            settings.mode = .apple; tab = .settings
        }
        #endif
        error = initialError
        recorder.onLimitReached = { [weak self] in self?.finishRecording() }
        recorder.onInterruption = { [weak self] in self?.stopRecordingForInterruption() }
        playback.onReleased = { [weak self] filename in self?.releasePlayedReply(filename) }
        startupTask = Task { [weak self] in await self?.initializeStorage() }
    }

    deinit {
        startupTask?.cancel(); settingsTask?.cancel(); pendingTurn?.cancel()
        pageTask?.cancel(); recordingTask?.cancel(); maintenanceTask?.cancel()
    }

    private func initializeStorage() async {
        defer {
            isPreparingStorage = false
            if isForeground { resumeSpeechPreparation() }
        }
        do {
            let page = try await archive.prepare(historyEnabled: settings.saveHistory)
            storageAvailable = true
            #if DEBUG
            // Explicit development launch action for the user's selected route;
            // preserve the existing service address, token and privacy settings.
            if ProcessInfo.processInfo.arguments.contains("-anyanwu-prefer-apple"),
               !ProcessInfo.processInfo.arguments.contains("-anyanwu-ui-testing") {
                settings.mode = .apple
                try await archive.saveSettings(settings)
            }
            #endif
            applyHistory(page)
            await refreshStorageUsage()
            resumeMaintenance()
            if ProcessInfo.processInfo.arguments.contains("-anyanwu-ui-testing"),
               ProcessInfo.processInfo.arguments.contains("-anyanwu-ui-chat-preview") {
                showConversation = true
                messages = [Message(role: .user, text: "Ndewo, kedu ka ị mere?"),
                            Message(role: .assistant, text: "Adị m mma, daalụ! Kedu maka gị?", isPreview: true),
                            Message(role: .user, text: "Kọọrọ m ihe gbasara aha gị."),
                            Message(role: .assistant, text: "Aha m bụ Anyanwụ.", isPreview: true)]
            }
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("-anyanwu-ui-testing"),
               ProcessInfo.processInfo.arguments.contains("-anyanwu-ui-font-preview") {
                showConversation = true
                messages = [Message(role: .user, text: "Ị Ọ Ụ Ṅ · ị ọ ụ ṅ"),
                            Message(role: .assistant, text: "Kedu ka ị mere? Adị m mma, daalụ!", isPreview: true),
                            Message(role: .user, text: "ị́ ị̀ ọ́ ọ̀ ụ́ ụ̀"),
                            Message(role: .assistant, text: "Ị́ Ị̀ Ọ́ Ọ̀ Ụ́ Ụ̀ Ṅ́", isPreview: true)]
            }
            #endif
        } catch { self.error = storageError(error) }
    }

    func waitForIdle() async {
        await startupTask?.value
        await settingsTask?.value
        await recordingTask?.value
        await pendingTurn?.value
        await pageTask?.value
    }

    @discardableResult func saveSettings(_ value: AssistantSettings) -> Task<Void, Never> {
        let historyChanged = settings.saveHistory != value.saveHistory
        if settings.mode != value.mode { cancelTurn(); cancelRecording(); stopPlayback(); appleAssistant.releaseResources() }
        if historyChanged && !value.saveHistory { newConversation(); conversations = [] }
        settings = value
        if isForeground && !isPreparingStorage { resumeSpeechPreparation() }
        let previous = settingsTask
        let startup = startupTask
        let archive = archive
        let currentMessages = messages
        let id = activeConversationID
        let createdAt = activeCreatedAt
        let task = Task { [weak self] in
            await startup?.value
            await previous?.value
            do {
                // Persist opt-out first; a crash before erasure is completed is
                // recovered by the same policy on the next launch.
                try await archive.saveSettings(value)
                try await archive.setHistoryEnabled(value.saveHistory)
                if historyChanged && value.saveHistory, let id { try await archive.saveSession(currentMessages, in: id, createdAt: createdAt) }
                guard let self else { return }
                self.storageAvailable = true
                await self.refreshHistory()
                await self.refreshStorageUsage()
            } catch { self?.error = self?.storageError(error) }
        }
        settingsTask = task
        return task
    }

    func newConversation() {
        let oldID = activeConversationID
        cancelTurn(); cancelRecording(); stopPlayback(); pageTask?.cancel()
        pageGeneration = UUID()
        activeConversationID = nil; activeCreatedAt = .now
        messages = []; hasOlderMessages = false; hasNewerMessages = false
        showConversation = false; isTextEntryVisible = false
        isLoadingConversation = false; isLoadingHistory = false
        draft = ""; error = nil; recordingNotice = nil; tab = .speak
        if let oldID {
            let archive = archive
            Task { try? await archive.clearTemporaryAudio(in: oldID) }
        }
    }

    func openComposer() { newConversation(); showConversation = true; isTextEntryVisible = true }

    func navigate(to destination: AppTab) {
        if destination == .speak { newConversation(); return }
        pageTask?.cancel(); pageGeneration = UUID()
        isLoadingConversation = false; isLoadingHistory = false
        clearTemporaryAudio()
        tab = destination
    }

    func open(_ conversation: ConversationSummary) {
        newConversation()
        activeConversationID = conversation.id
        activeCreatedAt = conversation.createdAt
        showConversation = true
        loadMessagePage()
    }

    func showOlderMessages() { if let id = messages.first?.id { loadMessagePage(before: id) } }
    func showNewerMessages() { if let id = messages.last?.id { loadMessagePage(after: id) } }
    func showOlderConversations() { if let cursor = conversations.last { loadHistoryPage(before: cursor) } }
    func showNewerConversations() { if let cursor = conversations.first { loadHistoryPage(after: cursor) } }

    private func loadMessagePage(before: UUID? = nil, after: UUID? = nil) {
        guard retainsHistory, let id = activeConversationID, !isWorking else { return }
        pageTask?.cancel(); let generation = UUID(); pageGeneration = generation
        isLoadingConversation = true
        pageTask = Task { [weak self, archive] in
            do {
                let page = try await archive.messages(in: id, before: before, after: after)
                guard let self, !Task.isCancelled, self.pageGeneration == generation, self.activeConversationID == id else { return }
                self.messages = page.items; self.hasOlderMessages = page.hasOlder; self.hasNewerMessages = page.hasNewer
                self.isLoadingConversation = false
            } catch {
                guard let self, self.pageGeneration == generation else { return }
                self.error = self.storageError(error); self.isLoadingConversation = false
            }
        }
    }

    private func loadHistoryPage(before: ConversationSummary? = nil, after: ConversationSummary? = nil) {
        guard retainsHistory else { return }
        pageTask?.cancel(); let generation = UUID(); pageGeneration = generation
        isLoadingHistory = true
        pageTask = Task { [weak self, archive] in
            do {
                let page = try await archive.summaries(before: before, after: after)
                guard let self, !Task.isCancelled, self.pageGeneration == generation else { return }
                self.applyHistory(page); self.isLoadingHistory = false
            } catch {
                guard let self, self.pageGeneration == generation else { return }
                self.error = self.storageError(error); self.isLoadingHistory = false
            }
        }
    }

    private func applyHistory(_ page: ArchivePage<ConversationSummary>) {
        conversations = page.items; hasOlderConversations = page.hasOlder; hasNewerConversations = page.hasNewer
    }

    private func refreshHistory() async {
        do { applyHistory(try await archive.summaries()) }
        catch { self.error = storageError(error) }
    }

    func sendText(_ value: String? = nil) {
        let text = (value ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isWorking, !isPreparingStorage, !isLoadingConversation,
              !isStartingRecording, !recorder.isRecording else { return }
        guard text.count <= 8_000, text.utf8.count <= 100_000 else {
            error = language.text("Please keep each message under 8,000 characters.", "Biko mee ka ozi ọ bụla dị n'okpuru mkpụrụedemede 8,000."); return
        }
        guard canAddTurn() else { return }
        draft = ""
        beginTurn(message: Message(role: .user, text: text), audio: nil, retry: false)
    }

    private func canAddTurn() -> Bool {
        if !retainsHistory && messages.count > ResourcePolicy.maximumSessionMessages - 2 {
            error = language.text("Start a new conversation to continue. History is off, so this session has a memory limit.", "Malite mkparịta ụka ọhụrụ. Agbanyụọla akụkọ, ya mere ohere mkparịta ụka a nwere oke.")
            return false
        }
        return true
    }

    func toggleRecording() {
        guard !isWorking, !isPreparingStorage, !isLoadingConversation else { return }
        if recorder.isRecording { finishRecording(); return }
        guard recordingTask == nil, canAddTurn() else { return }
        if settings.mode == .apple && (appleAssistant.isInstalling || !appleAssistant.capabilities.speechInstalled) {
            showConversation = true; isTextEntryVisible = false
            error = nil; recordingNotice = nil
            resumeSpeechPreparation()
            return
        }
        if activeConversationID == nil { activeConversationID = UUID(); activeCreatedAt = .now }
        guard let id = activeConversationID else { return }
        stopPlayback(); error = nil; recordingNotice = nil
        showConversation = true; isTextEntryVisible = false
        recordingTranscript = ""; isStartingRecording = true
        let generation = UUID(); recordingGeneration = generation
        let archive = archive, recorder = recorder
        recordingTask = Task { [weak self] in
            var url: URL?
            do {
                if self?.settings.mode == .apple { try await self?.appleAssistant.requireRecognition() }
                try Task.checkCancellation()
                url = try await archive.recordingURL(in: id)
                try Task.checkCancellation()
                try await recorder.start(at: url!)
                guard let self, self.recordingGeneration == generation else { recorder.cancel(); return }
                self.recordingURL = url; self.recordingTask = nil; self.isStartingRecording = false
            } catch {
                if let url { _ = try? await archive.removeTemporaryAudio(url.lastPathComponent, in: id) }
                guard let self, self.recordingGeneration == generation else { return }
                self.recordingTask = nil; self.isStartingRecording = false
                if !(error is CancellationError) { self.error = self.localizedError(error) }
            }
        }
    }

    /// Receives provisional words from a future streaming ASR connection.
    /// Session identity prevents late results from changing a different recording.
    /// Partial text is UI-only; only the final service transcript is archived.
    func receivePartialTranscript(_ text: String, sessionID: UUID) {
        guard recorder.isRecording, recordingGeneration == sessionID,
              text.count <= 8_000, text.utf8.count <= 100_000 else { return }
        recordingTranscript = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func finishRecording() {
        let duration = recorder.elapsed
        guard let url = recorder.stop(), let id = activeConversationID else { return }
        recordingURL = nil
        guard duration >= 0.5 else {
            recordingTranscript = ""
            let archive = archive
            Task { try? await archive.removeTemporaryAudio(url.lastPathComponent, in: id) }
            recordingNotice = language.text("That was a little short. Tap the microphone and try again.", "Ndekọ ahụ dị mkpụmkpụ. Pịa igwe okwu, wee nwaa ọzọ.")
            return
        }
        beginTurn(message: Message(role: .user, text: "Voice message · \(Int(duration))s", audioFilename: url.lastPathComponent), audio: url, retry: false)
    }

    func stopRecordingForInterruption() {
        let wasRecording = recorder.isRecording
        cancelRecording()
        if wasRecording { recordingNotice = language.text("Recording stopped. Tap the microphone when you’re ready.", "Akwụsịla ndekọ. Pịa igwe okwu mgbe ị dị njikere.") }
    }

    func cancelRecording() {
        recordingGeneration = UUID(); recordingTask?.cancel(); recordingTask = nil
        isStartingRecording = false; recordingTranscript = ""
        recorder.cancel(); recordingURL = nil
    }

    func cancelTurn() {
        turnGeneration = UUID(); pendingTurn?.cancel(); pendingTurn = nil
        isWorking = false; inputFilename = nil; transcribingMessageID = nil; recordingTranscript = ""
    }

    private func beginTurn(message: Message, audio: URL?, retry: Bool) {
        if activeConversationID == nil { activeConversationID = UUID(); activeCreatedAt = .now }
        guard let id = activeConversationID else { return }
        error = nil; recordingNotice = nil; showConversation = true
        let generation = UUID(); turnGeneration = generation
        isWorking = true; inputFilename = audio?.lastPathComponent
        transcribingMessageID = audio == nil ? nil : message.id
        let preferences = settings, createdAt = activeCreatedAt
        pendingTurn = Task { [weak self] in
            guard let self else { return }
            defer {
                if self.turnGeneration == generation {
                    self.isWorking = false; self.inputFilename = nil; self.pendingTurn = nil
                    self.transcribingMessageID = nil; self.recordingTranscript = ""
                }
            }
            await self.startupTask?.value
            await self.settingsTask?.value
            do {
                try Task.checkCancellation()
                if self.hasNewerMessages && self.retainsHistory {
                    let page = try await self.archive.messages(in: id)
                    guard self.isCurrent(generation, id) else { return }
                    self.messages = page.items; self.hasOlderMessages = page.hasOlder; self.hasNewerMessages = false
                }
                if !retry { await self.append(message, in: id, createdAt: createdAt) }
                guard self.isCurrent(generation, id) else { return }
                if preferences.mode == .apple {
                    let context = self.retainsHistory
                        ? try await self.archive.context(in: id, excluding: message.id)
                        : ContextWindow.messages(from: self.messages, excluding: message.id)
                    let text: String
                    if let audio {
                        text = try await self.appleAssistant.transcribe(audio)
                        try Task.checkCancellation()
                        guard self.isCurrent(generation, id) else { return }
                        var transcribed = message
                        transcribed.text = text; transcribed.audioFilename = nil
                        if let index = self.messages.firstIndex(where: { $0.id == message.id }) { self.messages[index] = transcribed }
                        self.transcribingMessageID = nil; self.recordingTranscript = ""
                        await self.persist(transcribed, in: id, createdAt: createdAt)
                        try await self.archive.removeTemporaryAudio(audio.lastPathComponent, in: id)
                    } else { text = message.text }
                    let reply = try await self.appleAssistant.reply(to: text, context: context, language: preferences.interfaceLanguage)
                    guard self.isCurrent(generation, id) else { return }
                    let output = Message(role: .assistant, text: reply.text, isPreview: reply.isPreview)
                    await self.append(output, in: id, createdAt: createdAt)
                    guard self.isCurrent(generation, id) else { return }
                    self.recordingNotice = reply.notice
                    if preferences.autoplayReplies {
                        do { try self.appleAssistant.speak(reply.text, messageID: output.id) }
                        catch {
                            let notice = AppleSpeechError.speechUnavailable.message(in: self.language)
                            self.recordingNotice = [reply.notice, notice].compactMap { $0 }.joined(separator: " ")
                        }
                    }
                    await self.refreshStorageUsage()
                    return
                }
                guard preferences.mode == .connected else {
                    if audio != nil {
                        self.recordingNotice = self.language.text("Connect a voice service to transcribe this recording.", "Jikọọ ọrụ olu ka e dee ndekọ a.")
                    } else {
                        await self.append(Message(role: .assistant, text: PreviewReplies.reply(to: message.text), isPreview: true), in: id, createdAt: createdAt)
                    }
                    return
                }
                let context = self.retainsHistory
                    ? try await self.archive.context(in: id, excluding: message.id)
                    : ContextWindow.messages(from: self.messages, excluding: message.id)
                let input: TurnRequest.Input
                if let audio {
                    let encoded = try await self.archive.encodedRecording(audio)
                    input = .init(type: "audio", audioBase64: encoded, mimeType: "audio/mp4")
                } else { input = .init(type: "text", text: message.text) }
                let response = try await self.service.turn(.init(conversationId: id.uuidString, interfaceLanguage: preferences.interfaceLanguage.rawValue, input: input, messages: context), baseURL: preferences.serverURL)
                try Task.checkCancellation()
                guard self.isCurrent(generation, id) else { return }
                if audio != nil {
                    guard let transcript = response.transcript, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AssistantServiceError.invalidResponse }
                    var transcribed = message
                    transcribed.text = transcript; transcribed.audioFilename = nil
                    if let index = self.messages.firstIndex(where: { $0.id == message.id }) { self.messages[index] = transcribed }
                    self.transcribingMessageID = nil; self.recordingTranscript = ""
                    await self.persist(transcribed, in: id, createdAt: createdAt)
                    if let filename = audio?.lastPathComponent { try await self.archive.removeTemporaryAudio(filename, in: id) }
                }
                var filename: String?
                if let encoded = response.audioBase64 {
                    do { filename = try await self.archive.saveReply(encoded, mimeType: response.audioMimeType, in: id) }
                    catch is CancellationError { throw CancellationError() }
                    catch { self.recordingNotice = self.language.text("Your reply is available as text; temporary audio could not be stored.", "Nzaghachi gị dị ka ederede; enweghị ike idowe ụda nwa oge.") }
                }
                guard self.isCurrent(generation, id) else {
                    if let filename { _ = try? await self.archive.removeTemporaryAudio(filename, in: id) }
                    return
                }
                await self.append(Message(role: .assistant, text: response.reply, isPreview: response.mode == "preview", audioFilename: filename), in: id, createdAt: createdAt)
                if let notice = response.notice { self.recordingNotice = notice }
                if preferences.autoplayReplies, let filename {
                    let url = try await self.archive.playbackURL(filename, in: id)
                    guard self.isCurrent(generation, id) else { return }
                    try self.playback.toggle(url: url)
                }
            } catch is CancellationError {
                // Buffers are released by the canceled HTTP operation. A failed
                // input clip remains temporary and can be retried until expiry.
            } catch {
                if self.isCurrent(generation, id) { self.error = self.localizedError(error) }
            }
            if self.isCurrent(generation, id) { await self.refreshStorageUsage() }
        }
    }

    private func isCurrent(_ generation: UUID, _ id: UUID) -> Bool {
        !Task.isCancelled && turnGeneration == generation && activeConversationID == id
    }

    private func append(_ message: Message, in id: UUID, createdAt: Date) async {
        guard activeConversationID == id, !Task.isCancelled else { return }
        messages.append(message)
        await persist(message, in: id, createdAt: createdAt)
        guard activeConversationID == id, !Task.isCancelled else { return }
        if retainsHistory && messages.count > ResourcePolicy.messagePageSize {
            messages.removeFirst(messages.count - ResourcePolicy.messagePageSize); hasOlderMessages = true
        }
    }

    private func persist(_ message: Message, in id: UUID, createdAt: Date) async {
        guard retainsHistory, !message.isVoicePlaceholder, !Task.isCancelled else { return }
        do { try await archive.save(message, in: id, createdAt: createdAt); await refreshHistory() }
        catch is CancellationError { }
        catch { storageAvailable = false; self.error = storageError(error) }
    }

    func retryVoice(_ message: Message) {
        guard !isWorking, settings.mode != .preview, message.isVoicePlaceholder,
              let id = activeConversationID, let filename = message.audioFilename else { return }
        pageTask = Task { [weak self, archive] in
            do {
                let url = try await archive.playbackURL(filename, in: id)
                guard let self, self.activeConversationID == id, !Task.isCancelled else { return }
                self.beginTurn(message: message, audio: url, retry: true)
            } catch { self?.error = self?.localizedError(error) }
        }
    }

    func togglePlayback(filename: String) {
        cancelRecording()
        appleAssistant.stopSpeaking()
        guard let id = activeConversationID else { return }
        pageTask = Task { [weak self, archive] in
            do {
                let url = try await archive.playbackURL(filename, in: id)
                guard let self, self.activeConversationID == id, !Task.isCancelled else { return }
                try self.playback.toggle(url: url)
            } catch { self?.error = self?.localizedError(error) }
        }
    }

    var isPlayingSpeech: Bool { playback.playingFilename != nil || appleAssistant.isSpeaking }
    func stopPlayback() { playback.stop(); appleAssistant.stopSpeaking() }

    func speakReply(_ message: Message) {
        guard message.role == .assistant, !isWorking else { return }
        if appleAssistant.speakingMessageID == message.id { appleAssistant.stopSpeaking(); return }
        cancelRecording(); playback.stop()
        do { try appleAssistant.speak(message.text, messageID: message.id) }
        catch { recordingNotice = AppleSpeechError.speechUnavailable.message(in: language) }
    }

    func removeOnDeviceSpeech() {
        cancelTurn(); cancelRecording(); stopPlayback()
        var value = settings; value.automaticallyPrepareSpeech = false
        let saved = saveSettings(value)
        Task {
            await saved.value
            await appleAssistant.removeSpeech(language: language)
        }
    }

    private func releasePlayedReply(_ filename: String) {
        guard filename.hasPrefix("temp-reply-"), let id = activeConversationID else { return }
        let archive = archive
        Task { [weak self] in
            do {
                try await archive.removeTemporaryAudio(filename, in: id)
                self?.removeAudioReferences([filename])
            } catch { self?.error = self?.storageError(error) }
        }
    }

    private func removeAudioReferences(_ filenames: Set<String>) {
        messages.removeAll { $0.isVoicePlaceholder && $0.audioFilename.map(filenames.contains) == true }
        for index in messages.indices where messages[index].audioFilename.map(filenames.contains) == true { messages[index].audioFilename = nil }
    }

    func deleteConversation(_ id: UUID) {
        if activeConversationID == id { newConversation() }
        pageTask = Task { [weak self, archive] in
            do { try await archive.deleteConversation(id); await self?.refreshHistory(); await self?.refreshStorageUsage() }
            catch { self?.error = self?.storageError(error) }
        }
    }

    func clearHistory() {
        newConversation(); conversations = []
        hasOlderConversations = false; hasNewerConversations = false
        pageTask = Task { [weak self, archive] in
            do { try await archive.eraseHistory(); await self?.refreshStorageUsage() }
            catch { self?.error = self?.storageError(error) }
        }
    }

    func clearTemporaryAudio() {
        cancelTurn(); cancelRecording(); stopPlayback()
        let filenames = Set(messages.compactMap(\.audioFilename).filter { $0.hasPrefix("temp-") })
        removeAudioReferences(filenames)
        pageTask = Task { [weak self, archive] in
            do { try await archive.clearTemporaryAudio(); await self?.refreshStorageUsage() }
            catch { self?.error = self?.storageError(error) }
        }
    }

    func refreshStorageUsage() async {
        do { storageUsage = try await archive.usage() }
        catch { /* A diagnostic read must not erase or reset the archive. */ }
    }

    func resumeMaintenance() {
        guard maintenanceTask == nil else { return }
        Task { [weak self, archive] in
            guard let self else { return }
            let protected = Set([self.recordingURL?.lastPathComponent, self.inputFilename, self.playback.playingFilename].compactMap { $0 })
            if let removed = try? await archive.removeExpiredAudio(protecting: protected) { self.removeAudioReferences(removed) }
        }
        maintenanceTask = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(30)) } catch { return }
                guard let self else { return }
                let protected = Set([self.recordingURL?.lastPathComponent, self.inputFilename, self.playback.playingFilename].compactMap { $0 })
                if let removed = try? await self.archive.removeExpiredAudio(protecting: protected) { self.removeAudioReferences(removed) }
            }
        }
    }

    func resumeForeground() {
        isForeground = true
        resumeMaintenance()
        if !isPreparingStorage { resumeSpeechPreparation() }
    }

    private func resumeSpeechPreparation() {
        appleAssistant.activatePreparation(automatically: settings.mode == .apple && settings.automaticallyPrepareSpeech,
                                           language: language)
    }

    func prepareSpeech() {
        var value = settings; value.automaticallyPrepareSpeech = true
        saveSettings(value)
        appleAssistant.installSpeech(language: language)
    }

    func cancelSpeechPreparation() {
        var value = settings; value.automaticallyPrepareSpeech = false
        saveSettings(value)
        appleAssistant.cancelInstall()
    }

    func suspendForBackground() {
        isForeground = false
        stopRecordingForInterruption(); cancelTurn(); stopPlayback(); appleAssistant.releaseResources()
        pageTask?.cancel(); pageGeneration = UUID()
        isLoadingConversation = false; isLoadingHistory = false
        maintenanceTask?.cancel(); maintenanceTask = nil
        handleMemoryPressure()
    }

    func handleMemoryPressure() {
        stopPlayback(); appleAssistant.releaseResources()
        if retainsHistory, messages.count > 20 {
            messages = Array(messages.suffix(20)); hasOlderMessages = true
        }
    }

    private func storageError(_ error: Error) -> String {
        language.text(error.localizedDescription, "Enweghị ike ịchekwa ma ọ bụ ịgụ akụkọ ugbu a. E chebere faịlụ gị.")
    }
    private func localizedError(_ error: Error) -> String {
        if let service = error as? AssistantServiceError { return service.message(in: language) }
        if let voice = error as? VoiceRecorder.VoiceError { return voice.message(in: language) }
        if let apple = error as? AppleSpeechError { return apple.message(in: language) }
        if error is ArchiveError { return storageError(error) }
        return language.text(error.localizedDescription, "Enweghị ike imecha ihe a. Biko nwaa ọzọ.")
    }
}
