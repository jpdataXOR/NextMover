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
def get_stock_data(stock_symbol, interval="1d"):
    global data_dic, current_values

    # Set appropriate period based on interval
    period = "1y" if interval == "1h" else "1y" if interval == "1d" else "max"
    
    try:
        instrument = yf.Ticker(stock_symbol)
        array_data = instrument.history(period=period, interval=interval, auto_adjust=False)
        
        if array_data.empty or len(array_data) < 5:
            return None
            
        # Calculate daily percent changes for our metrics
        array_data['Daily_Return'] = array_data['Close'].pct_change() * 100
        
        return array_data
    except Exception as e:
        st.warning(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Function to calculate days since last significant move
from datetime import timezone

def days_since_last_significant_move(data, threshold, direction="positive"):
    if direction == "positive":
        significant_days = data[data['Daily_Return'] >= threshold].index
    else:
        significant_days = data[data['Daily_Return'] <= threshold].index

    if not significant_days.empty:
        # Ensure both timestamps have the same timezone info
        now = datetime.now(tz=significant_days[-1].tz)
        return (now - significant_days[-1].to_pydatetime()).days

    return None

# Function to calculate metrics for a stock
def calculate_metrics(data, symbol):
    # Skip stocks with insufficient data
    if data is None or len(data) < 30:
        return None
    
    # Calculate average positive and negative moves
    positive_moves = data[data['Daily_Return'] > 0]['Daily_Return']
    negative_moves = data[data['Daily_Return'] < 0]['Daily_Return']
    
    avg_positive_move = positive_moves.mean() if len(positive_moves) > 0 else 0
    avg_negative_move = negative_moves.mean() if len(negative_moves) > 0 else 0
    
    # Calculate threshold for major moves (1.33 times average)
    positive_threshold = 1.33 * avg_positive_move
    negative_threshold = 1.33 * avg_negative_move
    
    # Find last major positive and negative moves
    last_major_positive = days_since_last_significant_move(data, positive_threshold, "positive")
    last_major_negative = days_since_last_significant_move(data, negative_threshold, "negative")
    
    # Calculate last 5 days performance
    last_5_days_perf = data['Close'].pct_change(5).iloc[-1] * 100 if len(data) >= 5 else None
    
    return {
        'Symbol': symbol,
        'Last_Major_Positive_Days': last_major_positive,
        'Last_Major_Negative_Days': last_major_negative,
        'Last_5_Days_Perf': last_5_days_perf,
        'Avg_Positive_Move': avg_positive_move,
        'Avg_Negative_Move': avg_negative_move,
        'Close': data['Close'].iloc[-1],
        'Volume': data['Volume'].iloc[-1]
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

# Main function
def main():
    st.title("NextMover - NASDAQ 100 Momentum Analyzer")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Option to limit number of stocks for testing
    test_mode = st.sidebar.checkbox("Test Mode (Limit to 10 stocks)", value=False)
    
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
                
                # Get stock data
                data = get_stock_data(symbol)
                if data is not None and not data.empty:
                    metrics = calculate_metrics(data, symbol)
                    if metrics:
                        results.append(metrics)
            
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
            else:
                st.error("No results found. Please check your internet connection or try again later.")
    
    # Display results if available (either from current analysis or previous run)
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        df_sorted = st.session_state.analysis_results
        
        # Display number of stocks analyzed
        st.write(f"Successfully analyzed {len(df_sorted)} NASDAQ-100 stocks")
        
        # Format DataFrame for display
        display_df = df_sorted.copy()
        
        # Format columns
        display_df['Last_5_Days_Perf'] = display_df['Last_5_Days_Perf'].round(2).astype(str) + '%'
        display_df['Avg_Positive_Move'] = display_df['Avg_Positive_Move'].round(2).astype(str) + '%'
        display_df['Avg_Negative_Move'] = display_df['Avg_Negative_Move'].round(2).astype(str) + '%'
        display_df['Close'] = display_df['Close'].round(2)
        display_df['Volume'] = display_df['Volume'].map('{:,.0f}'.format)
        display_df['Combined_Score'] = display_df['Combined_Score'].round(2)
        
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
        
        # Visualize top 10 stocks by score
        top10 = df_sorted.head(10).copy()
        
        # Create horizontal bar chart for top 10 stocks
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top10['Symbol'],
            x=top10['Combined_Score'],
            orientation='h',
            marker=dict(
                color=top10['Combined_Score'],
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
            # Get detailed data for selected stock
            stock_data = get_stock_data(selected_stock, interval="1d")
            
            if stock_data is not None and not stock_data.empty:
                # Create two columns for layout
                col1, col2 = st.columns(2)
                
                # Get metrics for selected stock
                stock_metrics = df_sorted[df_sorted['Symbol'] == selected_stock].iloc[0]
                
                # Display metrics in first column
                with col1:
                    st.write(f"### {selected_stock} Metrics")
                    st.write(f"Last Close: ${stock_metrics['Close']:.2f}")
                    st.write(f"Last 5 Days Performance: {stock_metrics['Last_5_Days_Perf']:.2f}%")
                    st.write(f"Last Major Positive Move: {stock_metrics['Last_Major_Positive_Days']} days ago")
                    st.write(f"Last Major Negative Move: {stock_metrics['Last_Major_Negative_Days']} days ago")
                    st.write(f"Average Positive Move: {stock_metrics['Avg_Positive_Move']:.2f}%")
                    st.write(f"Average Negative Move: {stock_metrics['Avg_Negative_Move']:.2f}%")
                    st.write(f"Combined Score: {stock_metrics['Combined_Score']:.2f}")
                
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
                
                # Add vertical lines for average positive and negative moves
                fig_hist.add_shape(
                    type="line",
                    x0=stock_metrics['Avg_Positive_Move'],
                    y0=0,
                    x1=stock_metrics['Avg_Positive_Move'],
                    y1=30,
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                fig_hist.add_shape(
                    type="line",
                    x0=stock_metrics['Avg_Negative_Move'],
                    y0=0,
                    x1=stock_metrics['Avg_Negative_Move'],
                    y1=30,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                # Add vertical lines for 1.33x moves
                fig_hist.add_shape(
                    type="line",
                    x0=1.33 * stock_metrics['Avg_Positive_Move'],
                    y0=0,
                    x1=1.33 * stock_metrics['Avg_Positive_Move'],
                    y1=30,
                    line=dict(color="darkgreen", width=2),
                )
                
                fig_hist.add_shape(
                    type="line",
                    x0=1.33 * stock_metrics['Avg_Negative_Move'],
                    y0=0,
                    x1=1.33 * stock_metrics['Avg_Negative_Move'],
                    y1=30,
                    line=dict(color="darkred", width=2),
                )
                
                fig_hist.update_layout(
                    title=f"{selected_stock} Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=300,
                    annotations=[
                        dict(
                            x=stock_metrics['Avg_Positive_Move'],
                            y=25,
                            text="Avg +",
                            showarrow=False,
                            font=dict(color="green")
                        ),
                        dict(
                            x=stock_metrics['Avg_Negative_Move'],
                            y=25,
                            text="Avg -",
                            showarrow=False,
                            font=dict(color="red")
                        ),
                        dict(
                            x=1.33 * stock_metrics['Avg_Positive_Move'],
                            y=20,
                            text="1.33x Avg +",
                            showarrow=False,
                            font=dict(color="darkgreen")
                        ),
                        dict(
                            x=1.33 * stock_metrics['Avg_Negative_Move'],
                            y=20,
                            text="1.33x Avg -",
                            showarrow=False,
                            font=dict(color="darkred")
                        )
                    ]
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()