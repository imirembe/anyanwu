import Foundation

/// A single attempt per foreground session or Wi-Fi reconnection. Explicit
/// cancellation/removal is persisted in settings and disables automatic setup.
struct SpeechPreparationPolicy {
    var enabled = false
    var foreground = false
    var wifiAvailable = false
    private(set) var attempted = false

    mutating func activate(enabled: Bool) {
        if !foreground { attempted = false }
        foreground = true
        self.enabled = enabled
    }

    mutating func updateNetwork(wifi: Bool) {
        if wifi && !wifiAvailable { attempted = false }
        wifiAvailable = wifi
    }

    mutating func suspend() { foreground = false; wifiAvailable = false }

    mutating func markAttempt() { attempted = true }

    mutating func claimAttempt(installed: Bool, installing: Bool, downloadsAllowed: Bool) -> Bool {
        guard enabled, foreground, wifiAvailable, !attempted, !installed, !installing, downloadsAllowed else { return false }
        attempted = true
        return true
    }
}
