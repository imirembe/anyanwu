import Foundation

/// Streams directly to an installation staging file, with no in-memory model
/// buffer. Each file has a pinned byte count; the installer also verifies SHA256.
extension BoundedHTTP {
    static func download(_ url: URL, to destination: URL, bytes: Int64,
                      configuration: URLSessionConfiguration = .ephemeral,
                      progress: @escaping @Sendable (Int64) -> Void) async throws {
        let operation = ModelFileTransfer(destination: destination, expected: bytes, configuration: configuration, progress: progress)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { operation.start(url, continuation: $0) }
        } onCancel: { operation.cancel() }
    }
}

private final class ModelFileTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let destination: URL
    private let expected: Int64
    private let configuration: URLSessionConfiguration
    private let progress: @Sendable (Int64) -> Void
    private var received: Int64 = 0
    private var reported: Int64 = 0
    private var file: FileHandle?
    private var session: URLSession?
    private var continuation: CheckedContinuation<Void, Error>?
    private var finished = false

    init(destination: URL, expected: Int64, configuration: URLSessionConfiguration, progress: @escaping @Sendable (Int64) -> Void) {
        self.destination = destination; self.expected = expected; self.configuration = configuration; self.progress = progress
        super.init()
    }

    func start(_ url: URL, continuation: CheckedContinuation<Void, Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); continuation.resume(throwing: CancellationError()); return }
        self.continuation = continuation
        guard url.scheme == "https", url.user == nil, url.password == nil, expected > 0,
              expected <= ResourcePolicy.onDeviceDownloadBytes else {
            lock.unlock(); finish(.failure(AppleSpeechError.invalidDownload)); return
        }
        do {
            var attributes: [FileAttributeKey: Any] = [:]
            #if os(iOS)
            attributes[.protectionKey] = FileProtectionType.complete
            #endif
            guard FileManager.default.createFile(atPath: destination.path, contents: nil, attributes: attributes) else {
                throw AppleSpeechError.invalidDownload
            }
            file = try FileHandle(forWritingTo: destination)
        } catch { lock.unlock(); finish(.failure(error)); return }
        configuration.urlCache = nil; configuration.httpCookieStorage = nil; configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 45; configuration.timeoutIntervalForResource = 900
        configuration.allowsExpensiveNetworkAccess = false
        configuration.allowsConstrainedNetworkAccess = false
        configuration.allowsCellularAccess = false
        let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        self.session = session
        var request = URLRequest(url: url)
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let task = session.dataTask(with: request)
        lock.unlock()
        task.resume()
    }

    func cancel() { finish(.failure(CancellationError())) }
    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let callback = continuation, session = session, file = file
        continuation = nil; self.session = nil; self.file = nil
        // Finish the file before waking the installer to hash or compile it.
        do { try file?.close() }
        catch {
            lock.unlock(); session?.invalidateAndCancel()
            try? FileManager.default.removeItem(at: destination)
            callback?.resume(throwing: error); return
        }
        if case .failure = result { try? FileManager.default.removeItem(at: destination) }
        lock.unlock()
        session?.invalidateAndCancel()
        callback?.resume(with: result)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let response = response as? HTTPURLResponse, response.statusCode == 200,
              response.expectedContentLength == -1 || response.expectedContentLength == expected else {
            completionHandler(.cancel); finish(.failure(AppleSpeechError.invalidDownload)); return
        }
        lock.lock(); let active = !finished; lock.unlock()
        completionHandler(active ? .allow : .cancel)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard request.url?.scheme == "https", request.url?.user == nil, request.url?.password == nil else {
            completionHandler(nil); finish(.failure(AppleSpeechError.invalidDownload)); return
        }
        completionHandler(request)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        guard Int64(data.count) <= expected - received else {
            lock.unlock(); finish(.failure(AppleSpeechError.invalidDownload)); return
        }
        do { try file?.write(contentsOf: data) }
        catch { lock.unlock(); finish(.failure(error)); return }
        received += Int64(data.count)
        let count = received
        let shouldReport = count == expected || count - reported >= 1_048_576
        if shouldReport { reported = count }
        lock.unlock()
        if shouldReport { progress(count) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)); return }
        lock.lock(); let complete = received == expected; lock.unlock()
        finish(complete ? .success(()) : .failure(AppleSpeechError.invalidDownload))
    }
}
