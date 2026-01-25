import codecs
import sys

path = "/media/zudva/git1/git/piper1-gpl/script/cutlist_review_ui.py"
with codecs.open(path, "r", "utf-8") as f:
    content = f.read()

# Construct the replacement string safely
script_code = r'''
    <script>
    function setupAudioSync() {
        // Poll because Gradio might replace elements dynamically
        setInterval(() => {
            const audio = document.querySelector('#main_audio audio') || document.querySelector('audio');
            const container = document.querySelector('.spectrogram-container');
            
            if (audio && container) {
                // Check if already hooked
                if (audio._isSyncHooked) return;
                
                const line = container.querySelector('.seek-line');
                const duration = parseFloat(container.dataset.duration);
                let animFrame;
                
                const update = () => {
                   const curr = audio.currentTime;
                   const d = duration || audio.duration;
                   if (line && d > 0) {
                       const pct = (curr / d) * 100;
                       line.style.left = pct + '%';
                       line.style.display = 'block';
                   }
                   if (!audio.paused && !audio.ended) {
                       animFrame = requestAnimationFrame(update);
                   }
                };
                
                audio.addEventListener('play', () => {
                    cancelAnimationFrame(animFrame);
                    update();
                });
                
                audio.addEventListener('pause', () => {
                    cancelAnimationFrame(animFrame);
                });
                
                audio.addEventListener('ended', () => {
                    cancelAnimationFrame(animFrame);
                });
                
                audio.addEventListener('seeked', update);
                
                audio._isSyncHooked = true;
                
                if (!audio.paused) update();
            }
        }, 500);
    }
    // Run on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupAudioSync);
    } else {
        setupAudioSync();
    }
    </script>
'''

# We need the assignment and the triple quotes exactly as in the file
# In the file it looks like:
#    js_sync = """
#    <script>
#    ...
#    </script>
#    """

new_js = '    js_sync = """' + script_code + '    """'

start_marker = 'js_sync = """'
end_marker = '    """'

p1 = content.split(start_marker)
if len(p1) < 2:
    print("Start marker not found")
    sys.exit(1)

pre = p1[0]
rest = p1[1]

end_idx = rest.find(end_marker)
if end_idx == -1:
    print("End marker not found")
    sys.exit(1)

post = rest[end_idx + len(end_marker):]

final_content = pre + new_js + post

with codecs.open(path, "w", "utf-8") as f:
    f.write(final_content)

print("Updated successfully")
