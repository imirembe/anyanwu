import Foundation

enum AppleSpeechError: LocalizedError {
    case physicalDevice, notInstalled, invalidModel, invalidDownload, insufficientSpace
    case busy, noSpeech, invalidAudio, speechUnavailable, replyUnavailable

    var errorDescription: String? { message(in: .english) }
    func message(in language: InterfaceLanguage) -> String {
        switch self {
        case .physicalDevice:
            language.text("Download speech recognition on a real iPhone. Model downloads are disabled in the simulator to keep large files off your Mac.", "Budata ọrụ ịghọta olu na iPhone n'ezie. A naghị ebudata nnukwu faịlụ na simulator.")
        case .notInstalled:
            language.text("Download Igbo speech recognition in Settings first. Your voice will then be transcribed on this iPhone.", "Buru ụzọ budata ọrụ ịghọta olu Igbo na Ntọala. Mgbe ahụ, a ga-ede olu gị na iPhone a.")
        case .invalidModel:
            language.text("The speech model could not be loaded. Remove it in Settings and download it again.", "Enweghị ike imepe ụdị ịghọta olu. Hichapụ ya na Ntọala, wee budata ya ọzọ.")
        case .invalidDownload:
            language.text("The speech download could not be verified. Please try again.", "Enweghị ike ịkwado faịlụ olu e budatara. Biko nwaa ọzọ.")
        case .insufficientSpace:
            language.text("Free at least 2 GB on this iPhone before downloading speech recognition.", "Mee ka ohere ruru 2 GB dị na iPhone a tupu ibudata ọrụ ịghọta olu.")
        case .busy:
            language.text("Speech setup is already running. Please wait or cancel it first.", "A na-ahazi ọrụ olu ugbu a. Biko chere ma ọ bụ kagbuo ya.")
        case .noSpeech:
            language.text("No clear speech was found. Please speak again.", "A nụghị okwu doro anya. Biko kwuo ọzọ.")
        case .invalidAudio:
            language.text("This recording could not be read. Keep recordings under 60 seconds and try again.", "Enweghị ike ịgụ ndekọ a. Mee ka ndekọ dị n'okpuru sekọnd iri isii, wee nwaa ọzọ.")
        case .speechUnavailable:
            language.text("An Igbo speaking voice is not available on this iPhone. Your reply is shown as text.", "Olu ịgụ Igbo adịghị na iPhone a. Nzaghachi gị dị ka ederede.")
        case .replyUnavailable:
            language.text("Apple’s reply model is unavailable for Igbo on this device. This is a prepared Igbo reply.", "Ụdị nzaghachi Apple anaghị akwado Igbo na ngwaọrụ a. Nke a bụ nzaghachi Igbo akwadoro.")
        }
    }
}

struct AppleCapabilities: Sendable {
    var speechInstalled = false
    var modelBytes: Int64 = 0
    var igboVoiceAvailable = false
    var igboGenerationAvailable = false
    var generationStatus = "Not checked"
    var isSimulator = false
}

struct AppleReply: Sendable {
    let text: String
    let isPreview: Bool
    let notice: String?
}
