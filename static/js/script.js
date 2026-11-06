document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-language-form] select').forEach((select) => {
    select.addEventListener('change', () => select.form.requestSubmit());
  });
  document.querySelectorAll('[data-password-toggle]').forEach((button) => {
    const input = document.getElementById(button.dataset.passwordToggle);
    if (!input) return;

    button.addEventListener('click', () => {
      const isHidden = input.type === 'password';
      input.type = isHidden ? 'text' : 'password';
      button.setAttribute('aria-label', isHidden ? 'Hide password' : 'Show password');
      button.setAttribute('aria-pressed', String(isHidden));
    });
  });

  document.querySelectorAll('[data-confirm]').forEach((form) => {
    form.addEventListener('submit', (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  document.querySelectorAll('[data-predict-form]').forEach((form) => {
    form.addEventListener('submit', () => {
      const loading = form.querySelector('[data-loading]');
      if (loading) loading.hidden = false;
      const button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
      }
    });
  });

  document.querySelectorAll('[data-translator]').forEach((widget) => {
    const message = (name) => widget.dataset[name] || '';
    const input = widget.querySelector('[data-translation-input]');
    const output = widget.querySelector('[data-translation-output]');
    const source = widget.querySelector('[data-source-language]');
    const target = widget.querySelector('[data-target-language]');
    const status = widget.querySelector('[data-translation-status]');
    const translateButton = widget.querySelector('[data-translate]');
    const copyButton = widget.querySelector('[data-copy-translation]');
    const speakButton = widget.querySelector('[data-speak-translation]');
    const inputCount = widget.querySelector('[data-input-count]');
    const outputCount = widget.querySelector('[data-output-count]');
    const history = widget.querySelector('[data-translation-history]');
    const languageNames = Object.fromEntries(
      [...source.options, ...target.options].map((option) => [option.value, option.textContent])
    );

    const updateInputCount = () => {
      const value = input.value;
      const words = value.trim() ? value.trim().split(/\s+/u).length : 0;
      const characters = [...value].length;
      inputCount.textContent = `${words} ${words === 1 ? message('wordLabel') : message('wordsLabel')} · ${characters}/${widget.dataset.maxChars} ${message('characterLabel')}`;
    };

    const clearResult = () => {
      output.textContent = output.dataset.placeholder;
      output.lang = target.value;
      output.classList.add('is-placeholder');
      outputCount.textContent = message('placeholderCount');
      copyButton.disabled = true;
      speakButton.disabled = true;
    };

    const showTranslation = (translated, from, to, createdAt, original) => {
      output.textContent = translated;
      output.lang = to;
      output.classList.remove('is-placeholder');
      const words = translated.trim() ? translated.trim().split(/\s+/u).length : 0;
      outputCount.textContent = `${words} ${words === 1 ? message('wordLabel') : message('wordsLabel')} · ${[...translated].length} ${message('characterLabel')}`;
      copyButton.disabled = false;
      speakButton.disabled = false;

      let empty = widget.querySelector('[data-history-empty]');
      if (empty) empty.remove();
      if (history) {
        const entry = document.createElement('article');
        entry.className = 'translation-history-item';
        const languages = document.createElement('div');
        languages.className = 'history-languages';
        const languagePair = document.createElement('span');
        languagePair.textContent = `${languageNames[from]} → ${languageNames[to]}`;
        const time = document.createElement('time');
        time.textContent = createdAt.replace('T', ' ').slice(0, 16);
        languages.append(languagePair, time);
        const originalText = document.createElement('p');
        originalText.lang = from;
        originalText.textContent = original;
        const resultText = document.createElement('p');
        resultText.className = 'history-result';
        resultText.lang = to;
        resultText.textContent = translated;
        entry.append(languages, originalText, resultText);
        history.prepend(entry);
        while (history.children.length > 10) history.lastElementChild.remove();
      }
    };

    const syncLanguageState = () => {
      input.lang = source.value;
      output.lang = target.value;
    };

    input.addEventListener('input', updateInputCount);
    source.addEventListener('change', syncLanguageState);
    target.addEventListener('change', () => {
      output.lang = target.value;
      if (!output.classList.contains('is-placeholder')) {
        outputCount.textContent = message('swappedCount');
      }
    });

    widget.querySelector('[data-clear-text]').addEventListener('click', () => {
      input.value = '';
      updateInputCount();
      clearResult();
      status.textContent = '';
      input.focus();
    });

    widget.querySelector('[data-swap-languages]').addEventListener('click', () => {
      const oldSource = source.value;
      source.value = target.value;
      target.value = oldSource;
      syncLanguageState();
      if (!output.classList.contains('is-placeholder')) {
        const oldInput = input.value;
        input.value = output.textContent;
        output.textContent = oldInput || output.dataset.placeholder;
        output.classList.toggle('is-placeholder', !oldInput);
        output.lang = target.value;
        outputCount.textContent = oldInput ? message('swappedCount') : message('placeholderCount');
        copyButton.disabled = !oldInput;
        speakButton.disabled = !oldInput;
        updateInputCount();
      }
      status.textContent = '';
    });

    translateButton.addEventListener('click', async () => {
      if (!input.value.trim()) {
        status.textContent = message('emptyMessage');
        input.focus();
        return;
      }
      if (source.value === target.value) {
        status.textContent = message('sameMessage');
        return;
      }
      translateButton.disabled = true;
      translateButton.setAttribute('aria-busy', 'true');
      status.textContent = message('translatingMessage');
      try {
        const response = await fetch(widget.dataset.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': widget.dataset.csrfToken,
          },
          body: JSON.stringify({ text: input.value, source: source.value, target: target.value }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || message('genericError'));
        showTranslation(result.translated_text, result.source_language, result.target_language, result.created_at, input.value.trim());
        status.textContent = message('completeMessage');
      } catch (error) {
        status.textContent = error.message || message('genericError');
      } finally {
        translateButton.disabled = false;
        translateButton.removeAttribute('aria-busy');
      }
    });

    copyButton.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(output.textContent);
        status.textContent = message('copyMessage');
      } catch (_error) {
        status.textContent = message('copyError');
      }
    });

    speakButton.addEventListener('click', () => {
      if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
        status.textContent = message('readError');
        return;
      }
      const voice = window.speechSynthesis.getVoices().find((item) => item.lang.toLowerCase().startsWith(`${target.value.toLowerCase()}-`) || item.lang.toLowerCase() === target.value.toLowerCase());
      if (!voice) {
        status.textContent = `${message('voiceMissing')} ${languageNames[target.value]}.`;
        return;
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(output.textContent);
      utterance.lang = voice.lang;
      utterance.voice = voice;
      utterance.onerror = () => { status.textContent = message('voiceError'); };
      window.speechSynthesis.speak(utterance);
      status.textContent = `${message('readingLabel')} ${languageNames[target.value]}.`;
    });

    syncLanguageState();
    updateInputCount();
  });
});
