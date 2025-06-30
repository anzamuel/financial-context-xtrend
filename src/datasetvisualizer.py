import panel as pn
import hvplot.pandas
import holoviews as hv
import pandas as pd
import torch

from dataset import XTrendDataset, FEATURE_COLS

pn.extension('tabulator', sizing_mode="stretch_width")

def create_visualizer(dataset: XTrendDataset):

	# --- Tab A: Per-Asset Global Visualization ---

	unique_ticker_indices = torch.unique(dataset.context_s).tolist()
	available_tickers = [dataset.idx_to_ticker[i] for i in sorted(unique_ticker_indices)]
	ticker_select = pn.widgets.Select(name='Select Asset', options=available_tickers, value=available_tickers[0])

	@pn.depends(ticker_select.param.value)
	def plot_asset_data(ticker):
		ticker_idx = dataset.ticker_to_idx[ticker].item()
		price_df = dataset.price_data[ticker].to_pandas()

		price_plot = price_df.hvplot.line(
			x='date', y='close',
			responsive=True, height=400, grid=True,
			title=f"Price and Context Regimes for {ticker}"
		)

		context_mask = (dataset.context_s == ticker_idx)
		start_dates = pd.to_datetime(dataset.context_start_dates[context_mask].numpy(), unit='s')
		end_dates = pd.to_datetime(dataset.context_end_dates[context_mask].numpy(), unit='s')

		regime_spans = hv.Overlay()
		for start, end in zip(start_dates, end_dates):
			regime_spans *= hv.VSpan(start, end).opts(color='gray', alpha=0.2)

		return price_plot * regime_spans

	tab_a = pn.Column(
		pn.Row(ticker_select),
		plot_asset_data
	)

	# --- Tab B: Per-Sample Visualization ---

	mode_select = pn.widgets.RadioButtonGroup(name='Mode', options=['train', 'eval'], button_type='primary')
	idx_slider = pn.widgets.IntSlider(name='Sample Index', start=0, end=len(dataset) - 1, step=1, value=0)

	@pn.depends(mode_select.param.value, watch=True)
	def _update_slider_range(mode):
		dataset.set_mode(mode)
		idx_slider.end = len(dataset) - 1
		idx_slider.value = 0

	def create_tensor_viewer(tensor_data, title, columns):
		df = pd.DataFrame(tensor_data.numpy(), columns=columns)
		view_toggle = pn.widgets.RadioButtonGroup(name='View As', options=['Table', 'Plot'], value='Table', button_type='default')

		@pn.depends(view_toggle.param.value)
		def _get_view(view_type):
			if view_type == 'Table':
				return pn.pane.DataFrame(df, max_rows=10)
			else:
				return df.hvplot(responsive=True, height=250, grid=True, title=title).opts(yformatter='%.2f')

		return pn.Card(pn.Column(view_toggle, _get_view), title=title, collapsed=True)

	@pn.depends(mode_select.param.value, idx_slider.param.value)
	def view_sample(mode, idx):
		dataset.set_mode(mode)
		sample_with_meta = dataset.get_item_with_metadata(idx)

		metadata_pane = pn.pane.Markdown(f"""
			### Sample Metadata
			**Mode:** {mode}
			**Index:** {idx}
			**Target Ticker:** {sample_with_meta['metadata']['target_ticker']}
			**Target Start:** {sample_with_meta['metadata'].get('target_start_date', 'N/A')}
			**Target End:** {sample_with_meta['metadata'].get('target_end_date', 'N/A')}
		""")

		target_viewer = create_tensor_viewer(sample_with_meta['target_x'], "Target Features", FEATURE_COLS)

		context_viewers = []
		for i in range(dataset.context_sample_size):
			ctx_ticker = sample_with_meta['metadata']['context_tickers'][i]
			title = f"Context {i+1} (Ticker: {ctx_ticker})"
			context_viewer = create_tensor_viewer(sample_with_meta['context_x'][i], title, FEATURE_COLS)
			context_viewers.append(context_viewer)

		return pn.Column(
			metadata_pane,
			target_viewer,
			pn.pane.Markdown("### Sampled Contexts"),
			*context_viewers
		)

	tab_b = pn.Column(
		pn.Row(mode_select, idx_slider),
		view_sample
	)

	# --- Final Layout ---
	return pn.Tabs(
		('Global Asset View', tab_a),
		('Single Sample Inspector', tab_b),
		dynamic=True
	)

# if __name__ == "__main__":
# 	print("Initializing Dataset for Visualizer...")
# 	dataset = XTrendDataset(load_price_data=True)
# 	print("Dataset loaded.")

# 	dashboard = create_visualizer(dataset)
# 	dashboard.servable(title="XTrend Dataset Visualizer")

if __name__ == "__main__":
	# Create the simplest possible Panel object to display.
	hello_pane = pn.pane.Markdown("## Hello there")

	# Make this simple object servable.
	hello_pane.servable(title="Debug Test")
