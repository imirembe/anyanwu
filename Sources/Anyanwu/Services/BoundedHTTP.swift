import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Collects a bounded response on a serial URLSession delegate queue. A canceled
/// turn immediately drops the buffer and invalidates its cache-free session.
enum BoundedHTTP {
    static func data(for request: URLRequest, limit: Int,
                     configuration: URLSessionConfiguration = .ephemeral) async throws -> (Data, HTTPURLResponse) {
        let operation = HTTPCollection(limit: limit, configuration: configuration)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { operation.start(request, continuation: $0) }
        } onCancel: { operation.cancel() }
    }
}

private final class HTTPCollection: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private let configuration: URLSessionConfiguration
    private var session: URLSession?
    private var continuation: CheckedContinuation<(Data, HTTPURLResponse), Error>?
    private var response: HTTPURLResponse?
    private var buffer = Data()
    private var finished = false

    init(limit: Int, configuration: URLSessionConfiguration) {
        self.limit = limit
        self.configuration = configuration
        super.init()
    }

    func start(_ request: URLRequest, continuation: CheckedContinuation<(Data, HTTPURLResponse), Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); continuation.resume(throwing: CancellationError()); return }
        self.continuation = continuation
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForResource = request.timeoutInterval
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        self.session = session
        let task = session.dataTask(with: request)
        lock.unlock()
        task.resume()
    }

    func cancel() { finish(.failure(CancellationError())) }

    private func finish(_ result: Result<(Data, HTTPURLResponse), Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let callback = continuation, oldSession = session
        continuation = nil; session = nil; response = nil; buffer = Data()
        lock.unlock()
        oldSession?.invalidateAndCancel()
        callback?.resume(with: result)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel); finish(.failure(AssistantServiceError.invalidResponse)); return
        }
        guard response.expectedContentLength <= Int64(limit) else {
            completionHandler(.cancel); finish(.failure(AssistantServiceError.responseTooLarge)); return
        }
        lock.lock()
        let active = !finished
        if active { self.response = http }
        lock.unlock()
        completionHandler(active ? .allow : .cancel)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        guard data.count <= limit - buffer.count else {
            lock.unlock(); finish(.failure(AssistantServiceError.responseTooLarge)); return
        }
        buffer.append(data)
        lock.unlock()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)); return }
        lock.lock()
        let response = response, data = buffer
        lock.unlock()
        if let response { finish(.success((data, response))) }
        else { finish(.failure(AssistantServiceError.invalidResponse)) }
    }
}
