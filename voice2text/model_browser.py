"""Keyboard-first fuzzy browsing of compatible Hugging Face exports."""

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static
from textual.worker import get_current_worker

from .model_search import ModelCatalog, ModelSearchResult, discover_models, fuzzy_matches
from .models import IMPORT_MODEL_TYPES, ModelInfo, register_custom_model


class SearchResultItem(ListItem):
    def __init__(self, result: ModelSearchResult):
        self.result = result
        family = IMPORT_MODEL_TYPES[result.info.onnx_asr_name][0]
        languages = ", ".join(result.languages) if len(result.languages) <= 4 else f"{len(result.languages)} languages"
        super().__init__(Label(
            f"{result.info.repo_id}\n{result.downloads:,} downloads · {result.info.size_hint} · {family} · {languages or 'languages unspecified'}",
            markup=False,
        ))


class ModelSearchScreen(ModalScreen[ModelInfo | None]):
    CSS = """
    ModelSearchScreen { align: center middle; }
    #model-search-dialog {
        width: 95%; max-width: 100; height: 95%;
        border: thick $accent; background: $surface; padding: 0 1;
    }
    #model-search-title { text-style: bold; height: 1; }
    #model-search-status { height: 2; }
    #model-search-results { height: 1fr; min-height: 2; }
    SearchResultItem { height: 2; }
    SearchResultItem Label { width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    #model-search-details { height: 3; }
    #model-search-help { height: 2; color: $text-muted; }
    #model-search-buttons { height: 3; }
    #model-search-buttons Button { margin-right: 1; }
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("down,ctrl+n", "next_result", "Next", show=False, priority=True),
        Binding("up,ctrl+p", "previous_result", "Previous", show=False, priority=True),
        Binding("enter", "choose", "Choose", show=False, priority=True),
    ]

    def __init__(self):
        super().__init__()
        self._catalog: ModelCatalog | None = None
        self._loading = False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-search-dialog"):
            yield Static("Search compatible models — downloads ↓", id="model-search-title")
            yield Input(placeholder="Fuzzy filter: name, owner, model family, language…", id="model-search-query")
            yield Static("Connecting to Hugging Face…", id="model-search-status", markup=False)
            yield ListView(id="model-search-results")
            yield Static("↑/↓ or Ctrl-P/N to browse · Enter to add · Esc to close", id="model-search-details", markup=False)
            yield Static("Downloads measure popularity, not accuracy. Ultra is a good starting point; compare errors and speed on your own recordings.", id="model-search-help")
            with Horizontal(id="model-search-buttons"):
                yield Button("Add selected", id="choose-search-model", variant="primary", disabled=True)
                yield Button("Refresh", id="refresh-model-search", disabled=True)
                yield Button("Cancel", id="cancel-model-search")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._loading = True
        self._discover()

    @work(thread=True)
    def _discover(self, refresh: bool = False) -> None:
        app, worker = self.app, get_current_worker()

        def progress(message: str) -> None:
            if not worker.is_cancelled:
                app.call_from_thread(self._show_progress, message)

        try:
            catalog = discover_models(progress, lambda: worker.is_cancelled, refresh=refresh)
        except Exception as exc:
            if not worker.is_cancelled:
                app.call_from_thread(self._failed, str(exc))
        else:
            if not worker.is_cancelled:
                app.call_from_thread(self._loaded, catalog)

    def _show_progress(self, message: str) -> None:
        if self.is_mounted:
            self.query_one("#model-search-status", Static).update(message)

    def _failed(self, error: str) -> None:
        if self.is_mounted:
            self._loading = False
            self._show_progress(f"Search unavailable: {error}. Refresh to retry; paste-link import is also available.")
            self.query_one("#refresh-model-search", Button).disabled = False

    async def _loaded(self, catalog: ModelCatalog) -> None:
        if self.is_mounted:
            self._catalog = catalog
            self._loading = False
            self.query_one("#refresh-model-search", Button).disabled = False
            await self._filter()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-search-query" and self._catalog is not None:
            await self._filter()

    async def _filter(self) -> None:
        query = self.query_one(Input).value
        results = [result for result in self._catalog.results if fuzzy_matches(query, result.search_text)]
        listing = self.query_one(ListView)
        await listing.clear()
        await listing.extend(SearchResultItem(result) for result in results)
        listing.index = 0 if results else None
        self.query_one("#choose-search-model", Button).disabled = not results
        message = f"{len(results)}/{len(self._catalog.results)} compatible · {self._catalog.checked} candidates checked · downloads ↓"
        if not results:
            message += "\nNo matches. Clear filter or paste link."
            if self._catalog.unavailable:
                message += f" {self._catalog.unavailable} unavailable; refresh to retry."
        elif self._catalog.unavailable:
            message += f"\n{self._catalog.unavailable} unavailable repositories skipped; refresh to retry."
        self._show_progress(message)
        if not results:
            self.query_one("#model-search-details", Static).update("")

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, SearchResultItem):
            result = event.item.result
            languages = ", ".join(result.languages)
            if len(result.languages) > 7:
                languages = f"{len(result.languages)} languages ({', '.join(result.languages[:7])}, …)"
            self.query_one("#model-search-details", Static).update(
                f"https://huggingface.co/{result.info.repo_id}\n"
                f"Languages: {languages or 'not specified by publisher'}\n"
                "Layout checked; runtime compatibility is verified when loading."
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.action_choose()

    def action_next_result(self) -> None:
        self.query_one(ListView).action_cursor_down()

    def action_previous_result(self) -> None:
        self.query_one(ListView).action_cursor_up()

    def action_choose(self) -> None:
        item = self.query_one(ListView).highlighted_child
        if isinstance(item, SearchResultItem):
            try:
                info = register_custom_model(item.result.info)
            except Exception as exc:
                self._show_progress(str(exc))
            else:
                self.dismiss(info)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "choose-search-model":
            self.action_choose()
        elif event.button.id == "cancel-model-search":
            self.action_cancel()
        elif event.button.id == "refresh-model-search" and not self._loading:
            self._loading = True
            self.query_one("#refresh-model-search", Button).disabled = True
            self._discover(refresh=True)

    def action_cancel(self) -> None:
        self.dismiss(None)
