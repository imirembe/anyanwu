import unittest
from unittest.mock import patch
from backend.voice_research.bbc_probe import checked_url, cloud_guard
from backend.voice_research.bbc_samples import master_selection, playlist_window

BASE = "https://vod-hls-ww-live.akamaized.net/public/master.m3u8"


class BBCPilotTests(unittest.TestCase):
    def test_laptop_guard_precedes_network(self):
        with patch("backend.voice_research.bbc_probe.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "laptop"):
                cloud_guard()

    def test_only_observed_bbc_https_hosts(self):
        self.assertEqual(checked_url(BASE), BASE)
        for bad in ("http://vod-hls-ww-live.akamaized.net/a", "https://vod-hls-ww-live.akamaized.net.evil.example/a", "https://user:secret@vod-hls-ww-live.akamaized.net/a"):
            with self.assertRaises(RuntimeError):
                checked_url(bad)

    def test_original_audio_group_and_bounded_video(self):
        master = '''#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="original",DEFAULT=YES,URI="speech.m3u8",LANGUAGE="ig"
#EXT-X-STREAM-INF:BANDWIDTH=12000000,RESOLUTION=1080x1920
large.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=700000,RESOLUTION=360x640,AUDIO="original"
small.m3u8
'''
        video, audio, selected = master_selection(master, BASE)
        self.assertTrue(video.endswith("/small.m3u8"))
        self.assertTrue(audio.endswith("/speech.m3u8"))
        self.assertEqual(selected["resolution"], "360x640")

    def test_window_preserves_source_offset_and_init(self):
        playlist = '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n'+''.join(f'#EXTINF:6.0,\npart-{i}.m4s\n' for i in range(6))+'#EXT-X-ENDLIST\n'
        selected = playlist_window(playlist, BASE, start=13, duration=12)
        self.assertEqual([s["start"] for s in selected], [12,18,24])
        self.assertTrue(selected[0]["init"].endswith("/init.mp4"))

    def test_rejects_encrypted_live_and_unbounded_streams(self):
        for playlist in ('#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="key"\n#EXT-X-ENDLIST', '#EXTM3U\n#EXTINF:6,\na.ts', '#EXTM3U\n#EXT-X-BYTERANGE:100@0\n#EXT-X-ENDLIST'):
            with self.assertRaises(RuntimeError):
                playlist_window(playlist, BASE)
        with self.assertRaisesRegex(RuntimeError, "No bounded"):
            master_selection('#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=12000000,RESOLUTION=1080x1920\nhuge.m3u8', BASE)

    def test_new_twenty_second_window_excludes_prior_sample(self):
        playlist = '#EXTM3U\n'+''.join(f'#EXTINF:6.0,\npart-{i}.ts\n' for i in range(8))+'#EXT-X-ENDLIST\n'
        selected = playlist_window(playlist, BASE, start=24, duration=20)
        self.assertEqual([s['start'] for s in selected], [24,30,36,42])
        self.assertTrue(all(s['start']>=24 for s in selected))
        with self.assertRaisesRegex(RuntimeError,'not covered'):
            playlist_window(playlist, BASE, start=40, duration=20)


if __name__ == "__main__":
    unittest.main()
