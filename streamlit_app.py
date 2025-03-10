import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page title and layout
st.set_page_config(page_title="NextMover - NASDAQ 100 Momentum Analyzer", layout="wide")

# Global variables
data_dic = {}
current_values = []

# Function to load Nasdaq-100 symbols
@st.cache_data(ttl=3600)
def load_nasdaq100_symbols():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # The Nasdaq-100 components table is typically at index 4
    nasdaq_table = tables[4]
    # Use "Symbol" or "Ticker" column based on availability
    if "Symbol" in nasdaq_table.columns:
        symbols = nasdaq_table["Symbol"].tolist()
    elif "Ticker" in nasdaq_table.columns:
        symbols = nasdaq_table["Ticker"].tolist()
    else:
        st.error("Could not find the Symbol/Ticker column in the table.")
        symbols = []
    return symbols

# Function to get stock data using your format


def get_stock_data(stock_symbol, interval="1d", end_date=None):
    global data_dic, current_values

    try:
        instrument = yf.Ticker(stock_symbol)

        # Convert end_date to string format (YYYY-MM-DD) and ensure it's timezone-naive
        end_date_str = None
        start_date_str = None

        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.replace(tzinfo=None)  # Remove timezone
                end_date_str = end_date.strftime('%Y-%m-%d')
                start_date_str = (end_date - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years back
            elif isinstance(end_date, str):
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=None)  # Remove timezone
                end_date_str = end_date
                start_date_str = (end_date_dt - timedelta(days=5*365)).strftime('%Y-%m-%d')

        # Fetch historical data using start & end when end_date is provided
        if end_date_str:
            array_data = instrument.history(start=start_date_str, end=end_date_str, interval=interval, auto_adjust=False)
        else:
            period = "1y" if interval in ["1h", "1d"] else "max"
            array_data = instrument.history(period=period, interval=interval, auto_adjust=False)

        if array_data.empty or len(array_data) < 5:
            return None

        # Convert DataFrame index to timezone-naive format
        if isinstance(array_data.index, pd.DatetimeIndex):
            array_data.index = array_data.index.tz_localize(None)

        # Calculate daily percent changes
        array_data['Daily_Return'] = array_data['Close'].pct_change() * 100

        return array_data

    except Exception as e:
        st.warning(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Function to calculate days since last significant move
from datetime import timezone

def days_since_last_significant_move(data, threshold, direction="positive", reference_date=None):
    if reference_date is None:
        # If no reference date is provided, use the last date in the data
        if not data.empty:
            reference_date = data.index[-1].to_pydatetime()
        else:
            return None
    
    # Ensure reference_date is a datetime object        
    if isinstance(reference_date, pd.Timestamp):
        reference_date = reference_date.to_pydatetime()
            
    if direction == "positive":
        significant_days = data[data['Daily_Return'] >= threshold].index
    else:
        significant_days = data[data['Daily_Return'] <= threshold].index

    if not significant_days.empty:
        # Find the most recent significant day that is before or on the reference date
        valid_days = [day for day in significant_days if day.to_pydatetime() <= reference_date]
        if valid_days:
            most_recent = max(valid_days).to_pydatetime()
            # Ensure both timestamps have the same timezone info
            if most_recent.tzinfo and reference_date.tzinfo:
                return (reference_date - most_recent).days
            # If timezone info is missing, normalize both to UTC or remove timezone info
            # for simpler comparison
            if not most_recent.tzinfo and not reference_date.tzinfo:
                return (reference_date - most_recent).days
            # If one has timezone and other doesn't, normalize
            if not most_recent.tzinfo:
                most_recent = most_recent.replace(tzinfo=timezone.utc)
            if not reference_date.tzinfo:
                reference_date = reference_date.replace(tzinfo=timezone.utc)
            return (reference_date - most_recent).days

    return None

# Function to calculate metrics for a stock
def calculate_metrics(data, symbol, reference_date=None):
    # Skip stocks with insufficient data
    if data is None or len(data) < 30:
        return None
    
    # If reference_date is provided, filter data up to that date
    if reference_date:
        # Ensure reference_date is a pd.Timestamp for direct comparison
        if isinstance(reference_date, datetime):
            reference_date = pd.Timestamp(reference_date)
        data = data[data.index <= reference_date]
        if len(data) < 30:  # Ensure we still have enough data after filtering
            return None
    
    # Calculate average positive and negative moves
    positive_moves = data[data['Daily_Return'] > 0]['Daily_Return']
    negative_moves = data[data['Daily_Return'] < 0]['Daily_Return']
    
    avg_positive_move = positive_moves.mean() if len(positive_moves) > 0 else 0
    avg_negative_move = negative_moves.mean() if len(negative_moves) > 0 else 0
    
    # Calculate threshold for major moves (1.33 times average)
    positive_threshold = 1.33 * avg_positive_move
    negative_threshold = 1.33 * avg_negative_move
    
    # Get reference date for calculations
    ref_date = reference_date if reference_date else data.index[-1]
    if isinstance(ref_date, pd.Timestamp):
        ref_date_dt = ref_date.to_pydatetime()
    else:
        ref_date_dt = ref_date
    
    # Find last major positive and negative moves
    last_major_positive = days_since_last_significant_move(data, positive_threshold, "positive", ref_date_dt)
    last_major_negative = days_since_last_significant_move(data, negative_threshold, "negative", ref_date_dt)
    
    # Calculate last 5 days performance relative to the reference date
    if len(data) >= 5:
        # Find the index of the reference date in the DataFrame
        if reference_date:
            # Get the closest date in the index that's not after the reference date
            valid_dates = data.index[data.index <= ref_date]
            if not valid_dates.empty:
                ref_idx = valid_dates[-1]
                # Get data for the last 5 trading days up to ref_idx
                last_dates = data.index[-5:] if ref_idx == data.index[-1] else \
                            data.index[:data.index.get_loc(ref_idx) + 1][-5:]
                
                if len(last_dates) >= 5:
                    last_5_days_perf = (data.loc[last_dates[-1], 'Close'] / data.loc[last_dates[0], 'Close'] - 1) * 100
                else:
                    last_5_days_perf = None
            else:
                last_5_days_perf = None
        else:
            # Just use the last 5 days in the data
            last_5_days_perf = data['Close'].pct_change(5).iloc[-1] * 100 if len(data) >= 5 else None
    else:
        last_5_days_perf = None
    
    # Use the last available data point based on the reference date
    if not data.empty:
        close_price = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
    else:
        close_price = None
        volume = None
    
    return {
        'Symbol': symbol,
        'Last_Major_Positive_Days': last_major_positive,
        'Last_Major_Negative_Days': last_major_negative,
        'Last_5_Days_Perf': last_5_days_perf,
        'Avg_Positive_Move': avg_positive_move,
        'Avg_Negative_Move': avg_negative_move,
        'Close': close_price,
        'Volume': volume
    }

# Function to calculate combined score
def calculate_combined_score(row, weight_5days, weight_positive, weight_negative):
    # Skip rows with missing data
    if pd.isna(row['Last_Major_Positive_Days']) or pd.isna(row['Last_Major_Negative_Days']) or pd.isna(row['Last_5_Days_Perf']):
        return np.nan
    
    # Calculate combined score with weights
    score = (row['Last_5_Days_Perf'] * weight_5days) + \
            (row['Last_Major_Positive_Days'] * weight_positive) + \
            (row['Last_Major_Negative_Days'] * weight_negative)
    
    return score

# Helper function to print pattern data (required by your format)
def print_difference_data(arg_array, index, matched_length, forward_length):
    matched = [{
        'date': arg_array.iloc[count].name.strftime('%d-%b-%Y %H:%M'),
        'close': arg_array.iloc[count]['Close'],
        'percentage_difference': ((arg_array.iloc[count]['Close'] - arg_array.iloc[count+1]['Close']) /
                                  arg_array.iloc[count+1]['Close']) * 100
    } for count in range(index, index + matched_length)]

    indices = [{
        'date': arg_array.iloc[count].name.strftime('%d-%b-%Y %H:%M'),
        'close': arg_array.iloc[count]['Close'],
        'percentage_difference': ((arg_array.iloc[count-1]['Close'] - arg_array.iloc[count]['Close']) /
                                  arg_array.iloc[count]['Close']) * 100
    } for count in range(index, index - forward_length, -1)]

    future_average = sum(index['percentage_difference']
                         for index in indices) / len(indices)
    return indices, matched, future_average

# Helper function to get last valid trading day
def get_last_valid_trading_day(date):
    # Check if the provided date is a weekend
    if date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        # If weekend, adjust to previous Friday
        days_to_subtract = date.weekday() - 4  # 4 is Friday
        return date - timedelta(days=days_to_subtract)
    return date

# Main function
def main():
    st.title("NextMover - NASDAQ 100 Momentum Analyzer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Option to limit number of stocks for testing
    test_mode = st.sidebar.checkbox("Test Mode (Limit to 10 stocks)", value=False)
    
    # Date selector for historical analysis
    st.sidebar.subheader("Analysis Date")
    today = datetime.now().date()
    min_date = today - timedelta(days=365)  # Allow selection up to 1 year back
    
    # Add date picker for specific date
    selected_date = st.sidebar.date_input(
        "Select date for analysis (default: latest data)",
        value=today,
        min_value=min_date,
        max_value=today
    )
    
    # Adjust selected date to last valid trading day if it's a weekend
    selected_date = get_last_valid_trading_day(selected_date)
    
    # Option to use latest data or selected date
    use_latest_data = st.sidebar.checkbox("Use latest data", value=True)
    
    # Display the effective analysis date
    if use_latest_data:
        st.sidebar.info("Using latest available data for analysis")
        analysis_date = None
    else:
        # Convert date to datetime with time at market close (4:00 PM ET)
        analysis_datetime = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=16)
        st.sidebar.info(f"Analyzing data up to: {selected_date.strftime('%A, %B %d, %Y')}")
        analysis_date = analysis_datetime
    
    # Custom score weights
    st.sidebar.subheader("Score Weights")
    weight_5days = st.sidebar.slider("Last 5 Days Performance Weight", 0.0, 1.0, 0.5, 0.1)
    weight_positive = st.sidebar.slider("Last Major Positive Move Weight", -1.0, 0.0, -0.25, 0.05)
    weight_negative = st.sidebar.slider("Last Major Negative Move Weight", 0.0, 1.0, 0.25, 0.05)
    
    # Progress indicators
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Load Nasdaq-100 symbols
    with st.spinner('Loading NASDAQ-100 symbols...'):
        nasdaq_symbols = load_nasdaq100_symbols()
    
    if not nasdaq_symbols:
        st.error("Failed to load NASDAQ-100 symbols.")
        return
    
    # Limit symbols in test mode
    if test_mode:
        nasdaq_symbols = nasdaq_symbols[:10]
        st.info("Test mode: Using only the first 10 NASDAQ-100 stocks")
    
    # Fetch data and calculate metrics for all symbols
    if st.button("Analyze NASDAQ-100 Stocks"):
        with st.spinner('Analyzing NASDAQ-100 stocks...'):
            results = []
            for i, symbol in enumerate(nasdaq_symbols):
                # Update progress
                progress = (i + 1) / len(nasdaq_symbols)
                progress_bar.progress(progress)
                status_text.text(f"Processing {symbol}... ({i+1}/{len(nasdaq_symbols)})")
                
                try:
                    # Get stock data with optional end date
                    data = get_stock_data(symbol, end_date=analysis_date)
                    if data is not None and not data.empty:
                        metrics = calculate_metrics(data, symbol, analysis_date)
                        if metrics:
                            results.append(metrics)
                except Exception as e:
                    st.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Reset progress bar when done
            progress_bar.empty()
            status_text.empty()
            
            # Create DataFrame from results
            if results:
                df = pd.DataFrame(results)
                
                # Calculate combined score with user-defined weights
                df['Combined_Score'] = df.apply(
                    lambda row: calculate_combined_score(row, weight_5days, weight_positive, weight_negative), 
                    axis=1
                )
                
                # Sort by combined score (descending)
                df_sorted = df.sort_values('Combined_Score', ascending=False).reset_index(drop=True)
                
                # Store the results in session state for persistence
                st.session_state.analysis_results = df_sorted
                st.session_state.analysis_complete = True
                
                # Store analysis date in session state
                st.session_state.analysis_date = analysis_date
                st.session_state.use_latest_data = use_latest_data
            else:
                st.error("No results found. Please check your internet connection or try again later.")
    
    # Display results if available (either from current analysis or previous run)
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        df_sorted = st.session_state.analysis_results
        
        # Show analysis date information
        if hasattr(st.session_state, 'use_latest_data') and st.session_state.use_latest_data:
            st.info("Analysis based on latest available data")
        elif hasattr(st.session_state, 'analysis_date') and st.session_state.analysis_date:
            analysis_date_str = st.session_state.analysis_date.strftime('%A, %B %d, %Y')
            st.info(f"Analysis based on data up to: {analysis_date_str}")
        
        # Display number of stocks analyzed
        st.write(f"Successfully analyzed {len(df_sorted)} NASDAQ-100 stocks")
        
        # Format DataFrame for display
        display_df = df_sorted.copy()
        
        # Format columns - handle NaN values
        display_df['Last_5_Days_Perf'] = display_df['Last_5_Days_Perf'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        display_df['Avg_Positive_Move'] = display_df['Avg_Positive_Move'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        display_df['Avg_Negative_Move'] = display_df['Avg_Negative_Move'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        display_df['Close'] = display_df['Close'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        display_df['Volume'] = display_df['Volume'].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        display_df['Combined_Score'] = display_df['Combined_Score'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'Symbol': 'Stock',
            'Last_Major_Positive_Days': 'Last +1.33x Days Ago',
            'Last_Major_Negative_Days': 'Last -1.33x Days Ago',
            'Last_5_Days_Perf': 'Last 5 Days',
            'Avg_Positive_Move': 'Avg + Move',
            'Avg_Negative_Move': 'Avg - Move',
            'Combined_Score': 'Score'
        })
        
        # Display top stocks table
        st.subheader("Top NextMover Candidates")
        st.dataframe(display_df)
        
        # Visualize top 10 stocks by score - make sure to convert back to float for charts
        chart_df = df_sorted.head(10).copy()
        chart_df = chart_df.dropna(subset=['Combined_Score'])  # Drop rows with NaN scores
        
        if not chart_df.empty:
            # Create horizontal bar chart for top 10 stocks
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=chart_df['Symbol'],
                x=chart_df['Combined_Score'],
                orientation='h',
                marker=dict(
                    color=chart_df['Combined_Score'],
                    colorscale='Viridis'
                )
            ))
            
            fig.update_layout(
                title="Top 10 NextMover Candidates by Score",
                xaxis_title="Combined Score",
                yaxis_title="Stock Symbol",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add section for stock detail view
        st.subheader("Stock Details")
        selected_stock = st.selectbox("Select a stock to view details", df_sorted['Symbol'].tolist())
        
        if selected_stock:
            # Get the analysis date to use for historical data
            end_date = None
            if hasattr(st.session_state, 'analysis_date') and not st.session_state.use_latest_data:
                end_date = st.session_state.analysis_date
            
            # Get detailed data for selected stock with optional end date
            stock_data = get_stock_data(selected_stock, interval="1d", end_date=end_date)
            
            if stock_data is not None and not stock_data.empty:
                # Create two columns for layout
                col1, col2 = st.columns(2)
                
                # Get metrics for selected stock
                stock_metrics = df_sorted[df_sorted['Symbol'] == selected_stock].iloc[0]
                
                # Display metrics in first column
                with col1:
                    st.write(f"### {selected_stock} Metrics")
                    st.write(f"Last Close: ${stock_metrics['Close']:.2f}")
                    
                    # Handle potential NaN values
                    last_5_days = f"{stock_metrics['Last_5_Days_Perf']:.2f}%" if pd.notnull(stock_metrics['Last_5_Days_Perf']) else "N/A"
                    last_pos = f"{stock_metrics['Last_Major_Positive_Days']} days ago" if pd.notnull(stock_metrics['Last_Major_Positive_Days']) else "N/A"
                    last_neg = f"{stock_metrics['Last_Major_Negative_Days']} days ago" if pd.notnull(stock_metrics['Last_Major_Negative_Days']) else "N/A"
                    avg_pos = f"{stock_metrics['Avg_Positive_Move']:.2f}%" if pd.notnull(stock_metrics['Avg_Positive_Move']) else "N/A"
                    avg_neg = f"{stock_metrics['Avg_Negative_Move']:.2f}%" if pd.notnull(stock_metrics['Avg_Negative_Move']) else "N/A"
                    score = f"{stock_metrics['Combined_Score']:.2f}" if pd.notnull(stock_metrics['Combined_Score']) else "N/A"
                    
                    st.write(f"Last 5 Days Performance: {last_5_days}")
                    st.write(f"Last Major Positive Move: {last_pos}")
                    st.write(f"Last Major Negative Move: {last_neg}")
                    st.write(f"Average Positive Move: {avg_pos}")
                    st.write(f"Average Negative Move: {avg_neg}")
                    st.write(f"Combined Score: {score}")
                
                # Display chart in second column
                with col2:
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name="Price"
                    )])
                    
                    # Add a trace for the 20-day moving average
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=20).mean(),
                        mode='lines',
                        name='20-day MA',
                        line=dict(color='orange')
                    ))
                    
                    # Add a trace for the 50-day moving average
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(window=50).mean(),
                        mode='lines',
                        name='50-day MA',
                        line=dict(color='blue')
                    ))
                    
                    # Add vertical line for analysis date if using historical data
                    if end_date:
                        # Convert to string for plotting
                        analysis_date_str = end_date.strftime('%Y-%m-%d')
                        fig.add_vline(
                            x=analysis_date_str,
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Analysis Date"
                        )
                    
                    fig.update_layout(
                        title=f"{selected_stock} Price History (1 Year)",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display daily returns histogram
                st.subheader(f"{selected_stock} Daily Returns Distribution")
                
                # Create histogram of daily returns
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=stock_data['Daily_Return'].dropna(),
                    nbinsx=50,
                    marker_color='rgba(0, 128, 255, 0.7)'
                ))
                
                # Only add lines if we have valid metrics
                if pd.notnull(stock_metrics['Avg_Positive_Move']):
                    # Add vertical lines for average positive and negative moves
                    fig_hist.add_shape(
                        type="line",
                        x0=stock_metrics['Avg_Positive_Move'],
                        y0=0,
                        x1=stock_metrics['Avg_Positive_Move'],
                        y1=30,
                        line=dict(color="green", width=2, dash="dash"),
                    )
                    
                    # Add vertical lines for 1.33x positive move
                    fig_hist.add_shape(
                        type="line",
                        x0=1.33 * stock_metrics['Avg_Positive_Move'],
                        y0=0,
                        x1=1.33 * stock_metrics['Avg_Positive_Move'],
                        y1=30,
                        line=dict(color="darkgreen", width=2),
                    )
                    
                    # Add annotation for average positive move
                    fig_hist.add_annotation(
                        x=stock_metrics['Avg_Positive_Move'],
                        y=25,
                        text="Avg +",
                        showarrow=False,
                        font=dict(color="green")
                    )
                    
                    # Add annotation for 1.33x positive move
                    fig_hist.add_annotation(
                        x=1.33 * stock_metrics['Avg_Positive_Move'],
                        y=20,
                        text="1.33x Avg +",
                        showarrow=False,
                        font=dict(color="darkgreen")
                    )
                
                if pd.notnull(stock_metrics['Avg_Negative_Move']):
                    # Add vertical lines for average negative move
                    fig_hist.add_shape(
                        type="line",
                        x0=stock_metrics['Avg_Negative_Move'],
                        y0=0,
                        x1=stock_metrics['Avg_Negative_Move'],
                        y1=30,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    # Add vertical lines for 1.33x negative move
                    fig_hist.add_shape(
                        type="line",
                        x0=1.33 * stock_metrics['Avg_Negative_Move'],
                        y0=0,
                        x1=1.33 * stock_metrics['Avg_Negative_Move'],
                        y1=30,
                        line=dict(color="darkred", width=2),
                    )
                    
                    # Add annotation for average negative move
                    fig_hist.add_annotation(
                        x=stock_metrics['Avg_Negative_Move'],
                        y=25,
                        text="Avg -",
                        showarrow=False,
                        font=dict(color="red")
                    )
                    
                    # Add annotation for 1.33x negative move
                    fig_hist.add_annotation(
                        x=1.33 * stock_metrics['Avg_Negative_Move'],
                        y=20,
                        text="1.33x Avg -",
                        showarrow=False,
                        font=dict(color="darkred")
                    )
                
                fig_hist.update_layout(
                    title=f"{selected_stock} Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()