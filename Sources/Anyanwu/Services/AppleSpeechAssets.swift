import CoreML
import CryptoKit
import Foundation

actor AppleSpeechAssets {
    static let revision = "ce99f14bbc768d8dc6ffd3235a21cd9751ceb139"
    static let repository = "aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s"
    static let package = "omnilingual-ctc-300m-int8.mlpackage"
    static var downloadsAllowed: Bool {
        #if os(iOS) && !targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }
    struct FileSpec: Sendable {
        let path: String; let bytes: Int64; let sha256: String
        var url: URL { URL(string: "https://huggingface.co/\(AppleSpeechAssets.repository)/resolve/\(AppleSpeechAssets.revision)/\(path)")! }
    }
    static let files: [FileSpec] = [
        .init(path: "\(package)/Manifest.json", bytes: 617, sha256: "9ae1538cb39972b3e3734e5383564c099b875ac5e4e0508fe300b5d41793fd21"),
        .init(path: "\(package)/Data/com.apple.CoreML/model.mlmodel", bytes: 302477, sha256: "5208a9e4b58f82592bdef5177ea4a5ba44ad1034d143983ef39d257eb9ab0ece"),
        .init(path: "\(package)/Data/com.apple.CoreML/weights/weight.bin", bytes: 326444192, sha256: "158084272159ff8b0583846782beba8b5b118aa1255528b0a21d3a77c8061612"),
        .init(path: "tokenizer.model", bytes: 91481, sha256: "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e"),
        .init(path: "config.json", bytes: 482, sha256: "97a7fa4e25cd68c00b3543aaba47782bb9e60a145aedb8cd03b24d66f66a3bb8"),
        .init(path: "README.md", bytes: 4621, sha256: "1afe04664ffbc4aa64dbde825e8d9b0e6c717477dbe68a50743ad7e540f1a877")
    ]
    static var downloadBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }
    let root: URL
    private var prepared = false
    private var installing = false
    var activeDirectory: URL { root.appending(path: "active", directoryHint: .isDirectory) }

    init(directory: URL) { root = directory.appending(path: "OnDeviceSpeech", directoryHint: .isDirectory) }

    private func prepare() throws {
        guard !prepared else { return }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var url = root; try url.setResourceValues(values)
        // This directory owns models only. Never walk into history/audio data.
        for url in try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        where url.lastPathComponent.hasPrefix("staging-") { try? FileManager.default.removeItem(at: url) }
        prepared = true
    }

    func installed() throws -> Bool {
        try prepare()
        let marker = activeDirectory.appending(path: "revision.txt")
        guard (try? marker.resourceValues(forKeys: [.fileSizeKey]).fileSize) == Self.revision.utf8.count,
              (try? String(contentsOf: marker, encoding: .utf8)) == Self.revision,
              FileManager.default.fileExists(atPath: activeDirectory.appending(path: "Speech.mlmodelc").path),
              (try? activeDirectory.appending(path: "tokenizer.model").resourceValues(forKeys: [.fileSizeKey]).fileSize) == 91481
        else { return false }
        return true
    }

    func locations() throws -> (model: URL, vocabulary: URL) {
        guard try installed() else { throw AppleSpeechError.notInstalled }
        return (activeDirectory.appending(path: "Speech.mlmodelc"), activeDirectory.appending(path: "tokenizer.model"))
    }

    func usage() throws -> Int64 {
        try prepare()
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        var bytes: Int64 = 0
        while let url = files.nextObject() as? URL {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values.isRegularFile == true { bytes += Int64(values.fileSize ?? 0) }
        }
        return bytes
    }

    func install(progress: @escaping @Sendable (Double) -> Void) async throws {
        // The guard precedes all downloads and filesystem writes. Simulator
        // tests cannot inadvertently fetch the 327 MB model onto the laptop.
        guard Self.downloadsAllowed else { throw AppleSpeechError.physicalDevice }
        guard !installing else { throw AppleSpeechError.busy }
        try prepare()
        if try installed() { return }
        installing = true
        let staging = root.appending(path: "staging-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { installing = false; try? FileManager.default.removeItem(at: staging) }
        #if os(iOS)
        let available = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage ?? 0
        guard available >= ResourcePolicy.onDeviceInstallFreeBytes else { throw AppleSpeechError.insufficientSpace }
        #endif
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        var completed: Int64 = 0
        for spec in Self.files {
            try Task.checkCancellation()
            let target = staging.appending(path: spec.path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let base = completed
            try await BoundedHTTP.download(spec.url, to: target, bytes: spec.bytes) { count in
                progress(Double(base + count) / Double(Self.downloadBytes) * 0.9)
            }
            try Self.verify(target, spec: spec)
            completed += spec.bytes
        }
        try Task.checkCancellation()
        let tokenizer = staging.appending(path: "tokenizer.model")
        _ = try SentencePieceLexicon(data: Data(contentsOf: tokenizer))
        // Compile from the portable package on the target iPhone, rather than
        // trusting a model binary compiled for another OS/architecture.
        let compiled = try await MLModel.compileModel(at: staging.appending(path: Self.package))
        defer { try? FileManager.default.removeItem(at: compiled) }
        try Task.checkCancellation()
        guard try Self.directoryBytes(compiled) <= ResourcePolicy.onDeviceCompiledModelBytes else { throw AppleSpeechError.invalidModel }
        let ready = staging.appending(path: "ready", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: ready, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: compiled, to: ready.appending(path: "Speech.mlmodelc"))
        for name in ["tokenizer.model", "config.json", "README.md"] {
            try FileManager.default.moveItem(at: staging.appending(path: name), to: ready.appending(path: name))
        }
        try Data(Self.revision.utf8).write(to: ready.appending(path: "revision.txt"), options: .atomic)
        try Task.checkCancellation()
        if FileManager.default.fileExists(atPath: activeDirectory.path) { try FileManager.default.removeItem(at: activeDirectory) }
        try FileManager.default.moveItem(at: ready, to: activeDirectory)
        progress(1)
    }

    static func verify(_ file: URL, spec: FileSpec) throws {
        guard Int64(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) == spec.bytes else { throw AppleSpeechError.invalidDownload }
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 262_144), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == spec.sha256 else { throw AppleSpeechError.invalidDownload }
    }

    private static func directoryBytes(_ directory: URL) throws -> Int64 {
        guard let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { throw AppleSpeechError.invalidModel }
        var total: Int64 = 0
        while let file = files.nextObject() as? URL {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            if values.isRegularFile == true { total += Int64(values.fileSize ?? 0) }
            guard total <= ResourcePolicy.onDeviceCompiledModelBytes else { throw AppleSpeechError.invalidModel }
        }
        return total
    }

    func remove() throws {
        guard !installing else { throw AppleSpeechError.busy }
        try prepare()
        if FileManager.default.fileExists(atPath: activeDirectory.path) { try FileManager.default.removeItem(at: activeDirectory) }
    }
}
