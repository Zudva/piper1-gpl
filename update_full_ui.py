import codecs
import sys

path = "/media/zudva/git1/git/piper1-gpl/script/cutlist_review_ui.py"
with codecs.open(path, "r", "utf-8") as f:
    content = f.read()

# 1. Add 'import html'
if "import html" not in content:
    content = content.replace("import io", "import io\nimport html")

# 2. Update _run_transcription to use word_timestamps and build karaoke HTML
old_transcribe = '''            _log(f"Transcribing {target_file}...")
            model = model_cache["model"]
            segments, _ = model.transcribe(target_file, language="ru")
            text = " ".join([s.text for s in segments]).strip()
            _log(f"Transcription done. chars={len(text)}")
            return text'''

new_transcribe = '''            _log(f"Transcribing {target_file}...")
            model = model_cache["model"]
            # Use word_timestamps=True to get word-level timing
            segments, _ = model.transcribe(target_file, language="ru", word_timestamps=True)
            
            full_text_list = []
            html_parts = []
            html_parts.append('<div class="karaoke-text" style="line-height: 1.6; font-size: 1.1rem; padding: 10px; background: #f9fafb; border-radius: 4px;">')
            
            for segment in segments:
                for word in segment.words:
                    w = word.word.strip()
                    full_text_list.append(w)
                    safe_w = html.escape(w)
                    # Add data attributes for JS
                    html_parts.append(f'<span class="karaoke-word" data-start="{word.start:.2f}" data-end="{word.end:.2f}" style="margin-right: 4px; padding: 2px 4px; border-radius: 3px; transition: background 0.1s;">{safe_w}</span>')
            
            html_parts.append('</div>')
            
            text = " ".join(full_text_list).strip()
            karaoke = "".join(html_parts)
            
            _log(f"Transcription done. chars={len(text)}")
            return text, karaoke'''

content = content.replace(old_transcribe, new_transcribe)


# 3. Update return type of _run_transcription if needed? No, purely dynamic in Python.

# 4. Add karaoke_html component to layout
# Search for spectrogram_html definition
spectro_def = 'spectrogram_html = gr.HTML(label="Mel Spectrogram", elem_id="spectrogram_box")'
karaoke_def = '''spectrogram_html = gr.HTML(label="Mel Spectrogram", elem_id="spectrogram_box")
                
                with gr.Accordion("Karaoke / Alignment View", open=True):
                    karaoke_html = gr.HTML(label="Karaoke", value="<div style='color:gray'>Click 'Auto-Transcribe' to see word alignment here.</div>", elem_id="karaoke_box")'''

if "karaoke_html = gr.HTML" not in content:
    content = content.replace(spectro_def, karaoke_def)

# 5. Connect btn_transcribe to karaoke_html
# btn_transcribe.click(..., outputs=[text_input]) -> outputs=[text_input, karaoke_html]
btn_click = 'btn_transcribe.click(fn=_run_transcription, inputs=[audio_path_state, start_state, end_state], outputs=[text_input])'
btn_click_new = 'btn_transcribe.click(fn=_run_transcription, inputs=[audio_path_state, start_state, end_state], outputs=[text_input, karaoke_html])'
content = content.replace(btn_click, btn_click_new)

# 6. Update CSS to highlight active words
# Search for css = """
css_old = 'css = """'
css_new = '''css = """
    .active-word { background-color: #fde047; color: #000; font-weight: bold; box-shadow: 0 0 2px rgba(0,0,0,0.2); }
'''
content = content.replace(css_old, css_new)

# 7. Update JS Sync to handle karaoke
# We replace the entire setupAudioSync function content
new_js_code = r'''
    <script>
    function setupAudioSync() {
        // Poll because Gradio might replace elements dynamically
        setInterval(() => {
            const audio = document.querySelector('#main_audio audio') || document.querySelector('audio');
            const container = document.querySelector('.spectrogram-container');
            const karaokeBox = document.querySelector('#karaoke_box');
            
            if (audio) {
                // Determine if we need to hook (or re-hook if elements changed)
                const isHooked = audio._isSyncHooked;
                
                // If hooked, we still run the update loop via animation frame, 
                // but we also need to efficiently update karaoke even if we are "hooked".
                // Actually, the requestAnimationFrame loop handles it.
                
                if (isHooked) return;
                
                let animFrame;
                
                const update = () => {
                   if (audio.paused || audio.ended) return;

                   const curr = audio.currentTime;
                   
                   // 1. Spectrogram Sync
                   if (container && document.contains(container)) {
                       const line = container.querySelector('.seek-line');
                       const duration = parseFloat(container.dataset.duration);
                       const d = duration || audio.duration;
                       if (line && d > 0) {
                           const pct = (curr / d) * 100;
                           line.style.left = pct + '%';
                           line.style.display = 'block';
                       }
                   }
                   
                   // 2. Karaoke Sync
                   if (karaokeBox && document.contains(karaokeBox)) {
                       // Optimize: Select all spans once? No, Gradio replaces innerHTML on update.
                       // We can query selector inside the loop if list is small (~20 words).
                       const words = karaokeBox.querySelectorAll('.karaoke-word');
                       words.forEach(w => {
                           const start = parseFloat(w.dataset.start);
                           const end = parseFloat(w.dataset.end);
                           // Simple range check
                           if (curr >= start && curr <= end) {
                               w.classList.add('active-word');
                           } else {
                               w.classList.remove('active-word');
                           }
                       });
                   }

                   animFrame = requestAnimationFrame(update);
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
                
                audio.addEventListener('seeked', () => {
                   // One-shot update for seek
                   const curr = audio.currentTime;
                   // Spectrogram
                   if (container) {
                       const line = container.querySelector('.seek-line');
                       const duration = parseFloat(container.dataset.duration);
                       const d = duration || audio.duration;
                       if (line && d > 0) {
                           line.style.left = ((curr/d)*100) + '%';
                       }
                   }
                   // Karaoke
                   if (karaokeBox) {
                       const words = karaokeBox.querySelectorAll('.karaoke-word');
                       words.forEach(w => {
                           const start = parseFloat(w.dataset.start);
                           const end = parseFloat(w.dataset.end);
                           if (curr >= start && curr <= end) {
                               w.classList.add('active-word');
                           } else {
                               w.classList.remove('active-word');
                           }
                       });
                   }
                });
                
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

target_start_marker = 'js_sync = """'
target_end_marker = '    """'

p1 = content.split(target_start_marker)
if len(p1) == 2:
    pre = p1[0]
    rest = p1[1]
    end_idx = rest.find(target_end_marker)
    if end_idx != -1:
        post = rest[end_idx + len(target_end_marker):]
        # Replace JS block
        content = pre + 'js_sync = """' + new_js_code + '    """' + post

with codecs.open(path, "w", "utf-8") as f:
    f.write(content)
print("Updated full UI logic")
