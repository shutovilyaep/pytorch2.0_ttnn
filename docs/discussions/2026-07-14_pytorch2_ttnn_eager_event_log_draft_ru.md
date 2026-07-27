# Черновик (RU) - журнал событий: Eager execution, CI/build rescue `#1293`, gate silence, deprecation

**Статус:** подробный черновик. Публикация EN: `2026-07-14_pytorch2_ttnn_eager_event_log_final_en.md`.  
**Уровни:** **факт** / **записано** / **first-party** / **открыто**.

---
> EN final is authoritative for the 2026-07-27 green-CI verification on `254f2642`.


## За 60 секунд

1. На входе фокус звучал как **Eager execution** (+ «заодно посмотришь сборку»).
2. Фактически пришлось уходить в **rescue** сломанного build/CI/packaging на фоне дрейфа `tt-metal` API/headers/CMake - иначе Eager ops нечем стабильно гонять.
3. Линия `#1243` -> [`#1293`](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1293): ~41 commit / 54 files / +2980/−875; к **2025-11-20** ветка доведена до последней visible итерации; окно без конфликтов с main до **2025-12-08** (**факт**). Approving review / org-verified green - **не доказаны**.
4. В ту же неделю Joe закрывает Eager bounty `#1073` как «not needed» (**2025-11-17**), пока Eager PR line продолжается.
5. Дальше - **first-party**: отмены/опустошение регулярных созвонов, игнор в Slack, план внутренней группы про блокировку, затем резкий deprecation-сигнал; публично README deprecate `#1310` (2025-12-08).
6. После roll-off Joe формулирует причину как **implementation speed** на PyTorch + tt-metal refactor (**записано**, без скринов) - без PR/metric/owner.

---

## 1. Assignment framing vs фактический scope

- Staffing / контакт: сентябрь 2025 (дата staffing `{?}`; доказанный контакт с Joe **2025-09-12**).
- **First-party framing:** Eager ops - основная цель; build/CI - «по пути / начнёшь с job, которая версии обновляет».
- Публичный Eager demand: bounty [#1073](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1073) (Bert Large via `model.to("ttnn:0")`); также публичные инструкции/приоритизация ops (D1173 / `#1215`).

**Scope mismatch (совместимо, не умысел):** наняли/собеседовали вокруг Eager/ML, deliverable стал крупный build-system rescue.

---

## 2. Почему без Eager-сдвига утонул бы в tt-metal chase

**Факт среды:** `#1293` тянет submodule/Docker/sfpi/git-lfs, CMake/RPATH/wheel CI, правки под меняющийся Tensor/device API tt-metal.

**First-party:** авто-обновляющая версии CI job при уже сломанной сборке создавала «убегающее болото». Решение отвлечься от бесконечного pin-chase и всё же двигать Eager line **на той же fix-ветке** - иначе feature PR не имеют рабочей базы.

---

## 3. `#1243` -> `#1293`: что доставлено

| Когда | Событие | Уровень |
| --- | --- | --- |
| 2025-10-20 | `#1243` с fork | факт |
| 2025-11-12 | ready for review; старт visible commits rescue | факт |
| 2025-11-14 | Sala: `TT_METAL_HOME` deprecated; вопрос про удаление `add_subdirectory(tt-metal)` | записано |
| 2025-11-17 | Joe закрывает `#1073`: «Bounty not needed at this time» | записано |
| 2025-11-19 | `#1243` closed; `#1293` opened (тот же content); review Joe/Artem/Kevin | факт |
| 2025-11-20 | last visible commit; комментарий про correctness-before-optimize + dual test paths | факт + записано |
| 2025-11-21 | `#1296` Unary на `fix/tt_metal_bump` | факт |

Размер `#1293`: **41 commits, 54 files, +2980/−875**. Packaging -> `pyproject.toml` / scikit-build-core, wheel CI, docs BuildFlow - не «маленький CI hotfix».

### Что нельзя писать как факт

- «20 ноября осталось только Approve&Merge» - **частично**: mergeable/conflict-free окно доказано; approving review и agent-verified org green SHA - нет.
- Дословный EN Sala «можно сделать по-другому» - **не найден**; есть конкретные CMake/`TT_METAL_HOME` замечания + first-party ощущение блокирующе-размытого feedback.

---

## 4. Expectations / созвоны / игнор / «Блокируха» (first-party)

Помечать явно как **first-party**, без календарных скринов в этом пакете:

1. Созвон Joe+Artem про expectations: хочет **Eager к концу года / под Новый год**, чтобы хоть одна модель шла + оптимизации; люди уйдут в отпуска - лучше раньше. Точную дату **«10 ноября» не фиксируем** (в архиве нет calendar export).
2. После готовности ветки - отмены/«diff» регулярных 2x/week созвонов; заход в пустой Zoom; Slack «мы ещё работаем?» -> «да-да» -> игнор.
3. Эскалация внутри EPAM; план группы про блокировку + Technical Debt Discussion для несрочного.
4. Резкий call: проект **deprecated**; автор фиксировал тезисы текстом (экспорт чата здесь не приложен). Публичный якорь deprecate: [`#1310`](https://github.com/tenstorrent/pytorch2.0_ttnn/pull/1310) **2025-12-08**.

---

## 5. Roll-off и «implementation speed»

- Roll-off сигнал ~**2026-02-03** (chat exit; partial).
- **2026-02-06** Joe (verbatim, screenshots excluded): решение свелось к **implementation speed on the PyTorch project and the tt-metal refactor**; в том же сообщении - appreciate the work.
- PR/metric/owner/date в формулировке **нет**. **Записано как слова Joe**, не как engineering RCA.

Контр-факты для RCA «медленный contributor»:

- `#1293` остаётся Open на maintainer review gate.
- Eager PRs зависят от fix-ветки.
- Параллельно в `tt-metal` - серия merged migration PR (отдельный журнал).
- После deprecate/roll-off bounty model PR всё ещё мержились (напр. `#1335`, `#1280` Feb 2026). **Факт.**

---

## 6. Вопросы на review

1. Какой PR / check-run / metric поддерживал оценку «implementation speed»?
2. Кто был owner merge/review `#1293` после Nov 20 и почему не было approving review?
3. Как «Bounty not needed» (`#1073`, 17 Nov) сочетается с продолжением Eager line и later speed-label?
4. Когда scope письменно сменился с Eager feature на build rescue - и кто это принял?

---

## Исключено

Скрины Slack/calendar/LinkedIn, точная дата Nov 10, motive-as-fact, нерелевантное evening ASR.
