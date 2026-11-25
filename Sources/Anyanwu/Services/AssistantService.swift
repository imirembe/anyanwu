import Foundation

struct TurnRequest: Encodable, Sendable {
    struct Input: Encodable, Sendable {
        let type: String
        var text: String?
        var audioBase64: String?
        var mimeType: String?
    }
    struct ContextMessage: Encodable, Sendable {
        let role: String
        let content: String
    }
    let conversationId: String
    let inputLanguages = ["ig", "en"]
    let responseLanguage = "ig"
    let interfaceLanguage: String
    let input: Input
    let messages: [ContextMessage]
}

struct TurnResponse: Decodable, Sendable {
    let transcript: String?
    let reply: String
    let audioBase64: String?
    let audioMimeType: String?
    let mode: String
    let responseLanguage: String?
    let notice: String?
}

enum AssistantServiceError: LocalizedError {
    case invalidURL, invalidResponse, responseTooLarge, server(String)
    var errorDescription: String? {
        message(in: .english)
    }
    func message(in language: InterfaceLanguage) -> String {
        switch self {
        case .invalidURL: language.text("Enter an HTTPS service address. For the simulator, you can also use http://localhost:8787.", "Tinye adreesị HTTPS. Na simulator, ị nwekwara ike iji http://localhost:8787.")
        case .invalidResponse: language.text("Anyanwụ received an incomplete response. Please try again.", "Nzaghachi Anyanwụ ezughị ezu. Biko nwaa ọzọ.")
        case .responseTooLarge: language.text("The service returned too much data. Try a shorter message.", "Ọrụ ahụ weghachiri data karịrị oke. Nwaa ozi dị mkpụmkpụ.")
        case .server(let message): language.text(message, "Ọrụ olu enweghị ike imecha ihe a. Lelee njikọ ya na Ntọala, wee nwaa ọzọ.")
        }
    }
}

struct AssistantService: Sendable {
    static func validatedURL(_ raw: String) throws -> URL {
        guard let url = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)),
              let host = url.host, !host.isEmpty,
              url.user == nil, url.password == nil, url.query == nil, url.fragment == nil,
              url.scheme == "https" || (url.scheme == "http" && ["localhost", "127.0.0.1", "::1"].contains(host))
        else { throw AssistantServiceError.invalidURL }
        return url
    }

    func turn(_ body: TurnRequest, baseURL: String) async throws -> TurnResponse {
        try Task.checkCancellation()
        guard body.messages.count <= ResourcePolicy.contextMessageCount,
              body.messages.reduce(0, { $0 + $1.content.count }) <= ResourcePolicy.contextCharacterCount,
              (body.input.text?.count ?? 0) <= 8_000,
              (body.input.audioBase64?.utf8.count ?? 0) <= (ResourcePolicy.maximumAudioBytes + 2) / 3 * 4
        else { throw AssistantServiceError.responseTooLarge }
        let url = try Self.validatedURL(baseURL).appending(path: "v1/turns")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token = SecureToken.read(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONEncoder().encode(body)
        guard (request.httpBody?.count ?? 0) <= ResourcePolicy.maximumResponseBytes else { throw AssistantServiceError.responseTooLarge }
        let (data, response) = try await BoundedHTTP.data(for: request, limit: ResourcePolicy.maximumResponseBytes)
        guard (200..<300).contains(response.statusCode) else {
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            throw AssistantServiceError.server(String((object?["error"] as? String ?? "The voice service is unavailable. Please try again.").prefix(500)))
        }
        let result = try JSONDecoder().decode(TurnResponse.self, from: data)
        guard !result.reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              result.reply.count <= 20_000, result.reply.utf8.count <= 100_000,
              (result.transcript?.count ?? 0) <= 8_000, (result.transcript?.utf8.count ?? 0) <= 100_000,
              (result.audioBase64?.utf8.count ?? 0) <= (ResourcePolicy.maximumAudioBytes + 2) / 3 * 4,
              result.audioBase64 == nil || ["audio/mpeg", "audio/mp4", "audio/wav"].contains(result.audioMimeType ?? ""),
              (result.notice?.count ?? 0) <= 1_000,
              ["preview", "live"].contains(result.mode),
              result.responseLanguage == nil || result.responseLanguage == "ig"
        else { throw AssistantServiceError.invalidResponse }
        return result
    }

    func check(baseURL: String, interfaceLanguage: InterfaceLanguage = .english) async throws -> String {
        var request = URLRequest(url: try Self.validatedURL(baseURL).appending(path: "health"))
        request.timeoutInterval = 12
        if let token = SecureToken.read(), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, http) = try await BoundedHTTP.data(for: request, limit: 65_536)
        guard http.statusCode == 200,
              let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              payload["service"] as? String == "anyanwu"
        else { throw AssistantServiceError.server("Couldn’t connect to Anyanwụ. Check the address and access token.") }
        let capabilities = payload["capabilities"] as? [String: Bool] ?? [:]
        return capabilities["transcription"] == true
            ? interfaceLanguage.text("Connected · speech recognition available", "Ejikọtara · ọrụ ịghọta olu dị")
            : interfaceLanguage.text("Connected · preview available; speech recognition is not configured", "Ejikọtara · nlele dị; ejikọtaghị ọrụ ịghọta olu")
    }
}
