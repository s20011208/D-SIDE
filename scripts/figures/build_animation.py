"""Encode real panel screenshots as a compact, looping README demonstration."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, default=ROOT/'runs/animation-frames')
    args = parser.parse_args()
    recording = json.loads((args.frames/'frames.json').read_text())
    frames = [Image.open(args.frames/row['file']).convert('RGB') for row in recording['frames']]
    if len({im.size for im in frames}) != 1:
        raise ValueError('Recording frame dimensions must stay fixed')
    # One palette across all frames preserves small text without color flicker.
    sheet = Image.new('RGB', (frames[0].width, frames[0].height*4))
    for j, i in enumerate([0, len(frames)//3, 2*len(frames)//3, len(frames)-1]):
        sheet.paste(frames[i], (0, j*frames[0].height))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    output = ROOT/'assets/algorithm_demo.gif'
    indexed[0].save(output, save_all=True, append_images=indexed[1:],
                    duration=[row['duration_ms'] for row in recording['frames']],
                    loop=0, optimize=True, disposal=2)
    frames[0].save(ROOT/'assets/algorithm_demo.png', optimize=True)
    with Image.open(output) as gif:
        decoded_frames = gif.n_frames
        if decoded_frames < 2 or gif.info.get('loop') != 0:
            raise ValueError('Expected an infinitely looping animated GIF')
        duration = 0
        for i in range(decoded_frames):
            gif.seek(i); duration += gif.info['duration']
            if ImageChops.difference(gif.convert('RGB'), indexed[i].convert('RGB')).getbbox():
                raise ValueError('GIF frame %d differs from its full reference image' % i)
    manifest = dict(recording, image_size=list(frames[0].size),
                    encoded_frames=decoded_frames, duration_ms=duration,
                    gif_bytes=output.stat().st_size, decoded_frames_match_quantized_recording=True,
                    playback='Automatic looping demonstration; click the README image to interact.')
    (ROOT/'assets/algorithm_demo.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Built %s: %d frames, %.2f seconds, %.2f MiB' %
          (output.relative_to(ROOT), decoded_frames, duration/1000, output.stat().st_size/2**20))


if __name__ == '__main__':
    main()
