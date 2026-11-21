document.addEventListener('DOMContentLoaded', () => {
  // Start the forecast icon animations only after the page has been painted.
  document.querySelectorAll('[data-weather-icon]').forEach((icon) => {
    const supported = new Set(['sunny', 'partly-cloudy', 'cloudy', 'rainy', 'thunderstorm', 'snowy', 'unknown']);
    if (!supported.has(icon.dataset.condition)) return;
    window.requestAnimationFrame(() => icon.classList.add('is-ready'));
  });

  const districtSelect = document.getElementById('district');
  const districtFilter = document.getElementById('district-filter');
  const weatherForm = document.getElementById('prediction-form');
  const refreshDistrict = document.querySelector('[data-refresh-district]');
  const districtSuggestionList = document.getElementById('district-suggestions');
  const districtNoResults = document.querySelector('[data-district-no-results]');
  const openSuggestionsButton = document.querySelector('[data-open-district-suggestions]');
  if (districtSelect && weatherForm) {
    let navigationTimer = null;
    let navigationStarted = false;
    let activeSuggestion = -1;
    const districtSearchControl = districtFilter?.closest('.district-search-control');
    const districtOptions = [...districtSelect.options];
    const openDistrict = () => {
      if (navigationStarted) return;
      navigationStarted = true;
      const loading = document.querySelector('[data-loading]');
      if (loading) loading.hidden = false;
      if (districtSelect.value) {
        window.location.assign(`${weatherForm.action}?district=${encodeURIComponent(districtSelect.value)}`);
      }
    };
    const closeDistrictSuggestions = () => {
      if (!districtSuggestionList || !districtFilter) return;
      districtSuggestionList.hidden = true;
      districtFilter.setAttribute('aria-expanded', 'false');
      districtFilter.removeAttribute('aria-activedescendant');
      activeSuggestion = -1;
    };
    const renderDistrictSuggestions = (showAll = false) => {
      if (!districtFilter || !districtSuggestionList) return;
      const query = districtFilter.value.trim().toLocaleLowerCase();
      const matches = districtOptions.filter((option) =>
        showAll || `${option.textContent} ${option.value}`.toLocaleLowerCase().includes(query),
      );
      districtSuggestionList.replaceChildren();
      matches.forEach((option, index) => {
        const item = document.createElement('li');
        item.id = `district-option-${index}`;
        item.className = 'district-suggestion';
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', String(option.value === districtSelect.value));
        item.dataset.districtValue = option.value;
        item.textContent = option.textContent.trim();
        item.addEventListener('pointerdown', (event) => event.preventDefault());
        item.addEventListener('click', () => {
          districtSelect.value = option.value;
          districtFilter.value = option.textContent.trim();
          closeDistrictSuggestions();
          districtSelect.dispatchEvent(new Event('change', { bubbles: true }));
        });
        districtSuggestionList.append(item);
      });
      districtSuggestionList.hidden = false;
      districtFilter.setAttribute('aria-expanded', 'true');
      if (districtNoResults) districtNoResults.hidden = matches.length > 0;
    };
    const setActiveSuggestion = (nextIndex) => {
      if (!districtSuggestionList) return;
      const options = [...districtSuggestionList.querySelectorAll('[role="option"]')];
      if (!options.length) return;
      activeSuggestion = (nextIndex + options.length) % options.length;
      options.forEach((option, index) => option.classList.toggle('is-active', index === activeSuggestion));
      districtFilter?.setAttribute('aria-activedescendant', options[activeSuggestion].id);
      options[activeSuggestion].scrollIntoView?.({ block: 'nearest' });
    };
    if (districtFilter && districtSuggestionList) {
      districtFilter.addEventListener('focus', renderDistrictSuggestions);
      districtFilter.addEventListener('input', () => {
        activeSuggestion = -1;
        renderDistrictSuggestions();
      });
      districtFilter.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          if (districtSuggestionList.hidden) renderDistrictSuggestions();
          setActiveSuggestion(activeSuggestion + 1);
        } else if (event.key === 'ArrowUp') {
          event.preventDefault();
          if (districtSuggestionList.hidden) renderDistrictSuggestions();
          setActiveSuggestion(activeSuggestion < 0 ? districtSuggestionList.children.length - 1 : activeSuggestion - 1);
        } else if (event.key === 'Enter') {
          const options = [...districtSuggestionList.querySelectorAll('[role="option"]')];
          if (!options.length) {
            event.preventDefault();
            if (districtNoResults) districtNoResults.hidden = false;
            return;
          }
          event.preventDefault();
          options[activeSuggestion < 0 ? 0 : activeSuggestion].click();
        } else if (event.key === 'Escape') {
          closeDistrictSuggestions();
        }
      });
      openSuggestionsButton?.addEventListener('click', () => {
        districtFilter.focus();
        renderDistrictSuggestions(true);
      });
      districtSuggestionList.addEventListener('pointerdown', (event) => event.preventDefault());
      document.addEventListener('pointerdown', (event) => {
        if (districtSearchControl && !districtSearchControl.contains(event.target)) closeDistrictSuggestions();
      });
    }
    districtSelect.addEventListener('change', () => {
      if (navigationStarted) return;
      window.clearTimeout(navigationTimer);
      navigationTimer = window.setTimeout(openDistrict, 180);
    });
    if (refreshDistrict) refreshDistrict.addEventListener('click', openDistrict);
  }

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
