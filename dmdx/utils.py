def frames_to_wave(frames, hop_length=256):
    frames = frames.tolist()
    fl = len(frames[0])
    wave = frames[0]
    for frame in frames[1:]:
        l = len(wave)
        for i, sample in enumerate(frame[:-hop_length]):
            idx = l - (fl - hop_length) + i
            wave[idx] += sample
            wave[idx] / 2
        for sample in frame[-hop_length:]:
            wave.append(sample)
    return wave
