# Advanced Fibonacci-Gann Temporal Swing Trading Algorithm
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import math
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import talib

class AdvancedFibonacciGannTrader:
    """
    Ultra-advanced trading system combining:
    - Fibonacci time sequences (10,000 candle analysis)
    - Gann mechanical methods and square relationships
    - Multi-timeframe trend analysis
    - Temporal-spatial market geometry
    """
    
    def __init__(self, symbol: str = "EURUSD", primary_timeframe: int = mt5.TIMEFRAME_H1):
        self.symbol = symbol
        self.primary_tf = primary_timeframe
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.fib_sequence = self.generate_fibonacci_sequence(1000)
        
        # Enhanced parameters for 10,000 candle analysis
        self.historical_candles = 10000
        self.multi_tf_analysis = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
        
        # Gann mechanical method parameters
        self.gann_angles = [1, 2, 3, 4, 5, 6, 7, 8]  # Gann angle divisors
        self.gann_square_levels = [90, 144, 233, 377, 610, 987]  # Gann square numbers
        
        # Advanced trading parameters
        self.risk_percent = 1.5
        self.max_trades = 2
        self.risk_reward_ratio = 3.0
        self.trend_confirmation_period = 50
        
        # Market memory
        self.historical_data = {}
        self.gann_grid = {}
        self.fib_time_zones = []
        
        self.initialize_mt5()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup advanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'fibonacci_gann_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 with enhanced error handling"""
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        
        # Preload historical data
        self.load_historical_data()
        self.logger.info("MT5 initialized with 10,000 candle historical analysis")
        return True
    
    def load_historical_data(self):
        """Load extensive historical data for deep analysis"""
        self.logger.info("Loading 10,000 candles historical data...")
        
        for timeframe in self.multi_tf_analysis:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.historical_candles)
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                self.historical_data[timeframe] = df
                self.logger.info(f"Loaded {len(df)} candles for timeframe {timeframe}")
            
        # Initialize Gann grid
        self.initialize_gann_grid()
        
        # Calculate Fibonacci time zones
        self.calculate_fibonacci_time_zones()
    
    def initialize_gann_grid(self):
        """Initialize Gann mechanical grid based on historical extremes"""
        if mt5.TIMEFRAME_H1 not in self.historical_data:
            return
            
        df = self.historical_data[mt5.TIMEFRAME_H1]
        
        # Calculate Gann grid levels from historical data
        all_time_high = df['high'].max()
        all_time_low = df['low'].min()
        price_range = all_time_high - all_time_low
        
        # Gann square of 9 levels
        self.gann_grid = {
            'major_high': all_time_high,
            'major_low': all_time_low,
            'price_range': price_range,
            'square_levels': [],
            'angle_levels': []
        }
        
        # Calculate Gann square levels
        base_price = (all_time_high + all_time_low) / 2
        for level in self.gann_square_levels:
            square_factor = math.sqrt(level)
            gann_level = base_price * (1 + (square_factor - 1) / 8)
            self.gann_grid['square_levels'].append(gann_level)
        
        # Calculate Gann angle levels
        for angle in self.gann_angles:
            angle_price = base_price * (1 + (1/angle) * (price_range / base_price))
            self.gann_grid['angle_levels'].append(angle_price)
    
    def calculate_fibonacci_time_zones(self):
        """Calculate Fibonacci time zones from major market turns"""
        if mt5.TIMEFRAME_H4 not in self.historical_data:
            return
            
        df = self.historical_data[mt5.TIMEFRAME_H4]
        
        # Find major swing points in historical data
        swing_points = self.find_major_swing_points(df, period=100)
        
        # Calculate Fibonacci time projections from each major swing
        for i, swing in enumerate(swing_points[:-1]):
            current_swing = swing
            next_swing = swing_points[i + 1] if i + 1 < len(swing_points) else swing_points[-1]
            
            time_diff = (next_swing['time'] - current_swing['time']).total_seconds() / 3600  # hours
            
            # Project Fibonacci time zones forward
            for fib_num in [13, 21, 34, 55, 89, 144, 233]:
                fib_time = time_diff * fib_num / 55  # Normalize to Fibonacci 55
                projected_time = current_swing['time'] + timedelta(hours=fib_time)
                
                self.fib_time_zones.append({
                    'projection_time': projected_time,
                    'source_swing': current_swing,
                    'fibonacci_ratio': fib_num,
                    'strength': min(1.0, fib_num / 144.0)  # Strength based on Fibonacci size
                })
    
    def find_major_swing_points(self, df: pd.DataFrame, period: int = 100) -> List[Dict]:
        """Find major swing points in historical data"""
        swings = []
        
        # Use ZigZag indicator to find major swings
        high = df['high'].values
        low = df['low'].values
        
        # Simple swing detection algorithm
        for i in range(period, len(df) - period):
            # Check for swing high
            if (high[i] == max(high[i-period:i+period]) and 
                high[i] > np.mean(high[i-period:i+period]) * 1.02):
                
                swings.append({
                    'type': 'high',
                    'price': high[i],
                    'time': df.iloc[i]['time'],
                    'strength': high[i] / np.mean(high[i-period:i+period])
                })
            
            # Check for swing low
            if (low[i] == min(low[i-period:i+period]) and 
                low[i] < np.mean(low[i-period:i+period]) * 0.98):
                
                swings.append({
                    'type': 'low', 
                    'price': low[i],
                    'time': df.iloc[i]['time'],
                    'strength': np.mean(low[i-period:i+period]) / low[i]
                })
        
        return sorted(swings, key=lambda x: x['time'])
    
    def generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate extended Fibonacci sequence"""
        sequence = [0, 1]
        for i in range(2, n):
            next_fib = sequence[i-1] + sequence[i-2]
            sequence.append(next_fib)
        return sequence
    
    def analyze_multi_timeframe_trend(self) -> Dict:
        """Advanced multi-timeframe trend analysis"""
        trend_analysis = {}
        
        for timeframe in self.multi_tf_analysis:
            if timeframe in self.historical_data:
                df = self.historical_data[timeframe]
                trend_analysis[timeframe] = self.calculate_trend_strength(df)
        
        # Composite trend score
        composite_bullish = sum([analysis['bullish_strength'] for analysis in trend_analysis.values()])
        composite_bearish = sum([analysis['bearish_strength'] for analysis in trend_analysis.values()])
        
        return {
            'timeframe_analysis': trend_analysis,
            'composite_trend': 'BULLISH' if composite_bullish > composite_bearish else 'BEARISH',
            'trend_strength': abs(composite_bullish - composite_bearish) / len(trend_analysis),
            'alignment': self.calculate_timeframe_alignment(trend_analysis)
        }
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive trend strength using multiple indicators"""
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # EMA trend analysis
        ema_20 = talib.EMA(closes, timeperiod=20)
        ema_50 = talib.EMA(closes, timeperiod=50)
        ema_100 = talib.EMA(closes, timeperiod=100)
        
        # MACD trend momentum
        macd, macd_signal, macd_hist = talib.MACD(closes)
        
        # ADX trend strength
        adx = talib.ADX(highs, lows, closes, timeperiod=14)
        
        # Calculate bullish/bearish strength
        bullish_indicators = 0
        bearish_indicators = 0
        
        if len(closes) > 100:
            # EMA alignment
            if ema_20[-1] > ema_50[-1] > ema_100[-1]:
                bullish_indicators += 1
            elif ema_20[-1] < ema_50[-1] < ema_100[-1]:
                bearish_indicators += 1
            
            # MACD momentum
            if macd_hist[-1] > 0:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            # Price position relative to EMAs
            if closes[-1] > ema_20[-1]:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            # ADX trend strength
            if adx[-1] > 25:  Strong trend
                if bullish_indicators > bearish_indicators:
                    bullish_indicators += 1
                else:
                    bearish_indicators += 1
        
        total_indicators = max(bullish_indicators + bearish_indicators, 1)
        
        return {
            'bullish_strength': bullish_indicators / total_indicators,
            'bearish_strength': bearish_indicators / total_indicators,
            'adx_strength': adx[-1] if len(adx) > 0 else 0,
            'ema_alignment': 'BULLISH' if bullish_indicators > bearish_indicators else 'BEARISH'
        }
    
    def calculate_timeframe_alignment(self, trend_analysis: Dict) -> float:
        """Calculate how well timeframes are aligned in trend direction"""
        if not trend_analysis:
            return 0.0
        
        directions = []
        for analysis in trend_analysis.values():
            if analysis['bullish_strength'] > analysis['bearish_strength']:
                directions.append(1)  # Bullish
            else:
                directions.append(-1)  # Bearish
        
        # Calculate alignment (1.0 = perfect alignment, 0.0 = no alignment)
        alignment = sum(directions) / len(directions)
        return abs(alignment)
    
    def gann_mechanical_analysis(self, current_price: float) -> Dict:
        """Perform Gann mechanical method analysis"""
        if not self.gann_grid:
            return {}
        
        # Calculate distance to nearest Gann levels
        square_distances = [abs(level - current_price) for level in self.gann_grid['square_levels']]
        angle_distances = [abs(level - current_price) for level in self.gann_grid['angle_levels']]
        
        nearest_square = min(square_distances) if square_distances else 0
        nearest_angle = min(angle_distances) if angle_distances else 0
        
        # Calculate Gann square resistance/support
        square_resistance = min([level for level in self.gann_grid['square_levels'] if level > current_price], default=0)
        square_support = max([level for level in self.gann_grid['square_levels'] if level < current_price], default=0)
        
        return {
            'nearest_square_distance': nearest_square,
            'nearest_angle_distance': nearest_angle,
            'square_resistance': square_resistance,
            'square_support': square_support,
            'at_square_level': nearest_square < current_price * 0.001,  # Within 0.1%
            'at_angle_level': nearest_angle < current_price * 0.001,
            'square_strength': 1.0 - (nearest_square / current_price) if current_price > 0 else 0
        }
    
    def fibonacci_time_analysis(self) -> Dict:
        """Advanced Fibonacci time sequence analysis"""
        current_time = datetime.now()
        
        # Check Fibonacci time zones
        active_time_zones = []
        for zone in self.fib_time_zones:
            time_diff = (zone['projection_time'] - current_time).total_seconds() / 3600
            if abs(time_diff) < 24:  # Within 24 hours
                active_time_zones.append({
                    'zone': zone,
                    'hours_until': time_diff,
                    'strength': zone['strength'] * (1 - abs(time_diff) / 24)  # Strength decays with time
                })
        
        # Fibonacci time cycles
        total_minutes = int(current_time.timestamp() / 60)
        active_cycles = []
        
        for fib_num in [13, 21, 34, 55, 89, 144, 233]:
            cycle_phase = (total_minutes % fib_num) / fib_num
            cycle_strength = math.sin(cycle_phase * 2 * math.pi)
            
            # Check if near cycle turning point
            if abs(cycle_phase - 0.5) < 0.1 or cycle_phase < 0.1:  # Near cycle extremes
                active_cycles.append({
                    'fib_number': fib_num,
                    'phase': cycle_phase,
                    'strength': abs(cycle_strength),
                    'direction': 1 if cycle_strength > 0 else -1
                })
        
        return {
            'active_time_zones': active_time_zones,
            'active_cycles': active_cycles,
            'total_time_signals': len(active_time_zones) + len(active_cycles),
            'composite_time_strength': min(1.0, (len(active_time_zones) * 0.3 + len(active_cycles) * 0.2))
        }
    
    def spatial_temporal_confluence(self, current_price: float) -> Dict:
        """Analyze spatial-temporal confluence using Fibonacci and Gann"""
        time_analysis = self.fibonacci_time_analysis()
        gann_analysis = self.gann_mechanical_analysis(current_price)
        trend_analysis = self.analyze_multi_timeframe_trend()
        
        confluence_score = 0.0
        confluence_factors = []
        
        # Time zone confluence
        if time_analysis['active_time_zones']:
            max_zone_strength = max([zone['strength'] for zone in time_analysis['active_time_zones']])
            confluence_score += max_zone_strength * 0.4
            confluence_factors.append(f"TimeZone:{max_zone_strength:.2f}")
        
        # Gann level confluence
        if gann_analysis['at_square_level'] or gann_analysis['at_angle_level']:
            confluence_score += gann_analysis['square_strength'] * 0.3
            confluence_factors.append(f"GannLevel:{gann_analysis['square_strength']:.2f}")
        
        # Trend alignment confluence
        if trend_analysis['trend_strength'] > 0.6:
            confluence_score += trend_analysis['trend_strength'] * 0.3
            confluence_factors.append(f"Trend:{trend_analysis['trend_strength']:.2f}")
        
        # Time cycle confluence
        if time_analysis['active_cycles']:
            cycle_strength = sum([cycle['strength'] for cycle in time_analysis['active_cycles']]) / len(time_analysis['active_cycles'])
            confluence_score += cycle_strength * 0.2
            confluence_factors.append(f"Cycles:{cycle_strength:.2f}")
        
        return {
            'confluence_score': min(1.0, confluence_score),
            'factors': confluence_factors,
            'time_analysis': time_analysis,
            'gann_analysis': gann_analysis,
            'trend_analysis': trend_analysis
        }
    
    def calculate_dynamic_support_resistance(self) -> Dict:
        """Calculate dynamic support/resistance using Fibonacci and Gann"""
        if mt5.TIMEFRAME_H1 not in self.historical_data:
            return {}
        
        df = self.historical_data[mt5.TIMEFRAME_H1]
        current_price = df.iloc[-1]['close']
        
        # Fibonacci retracement levels from recent swing
        recent_high = df['high'].tail(100).max()
        recent_low = df['low'].tail(100).min()
        price_range = recent_high - recent_low
        
        fib_levels = {
            'fib_236': recent_high - price_range * 0.236,
            'fib_382': recent_high - price_range * 0.382,
            'fib_500': recent_high - price_range * 0.500,
            'fib_618': recent_high - price_range * 0.618,
            'fib_786': recent_high - price_range * 0.786
        }
        
        # Combine with Gann levels
        combined_levels = {}
        for name, level in fib_levels.items():
            combined_levels[name] = level
        
        # Add Gann square levels
        for i, gann_level in enumerate(self.gann_grid['square_levels']):
            if abs(gann_level - current_price) / current_price < 0.05:  # Within 5%
                combined_levels[f'gann_square_{i}'] = gann_level
        
        # Sort levels by proximity to current price
        sorted_levels = sorted(combined_levels.items(), key=lambda x: abs(x[1] - current_price))
        
        support_levels = [level for level in sorted_levels if level[1] < current_price][:3]
        resistance_levels = [level for level in sorted_levels if level[1] > current_price][:3]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'current_price': current_price,
            'price_position': (current_price - recent_low) / price_range if price_range > 0 else 0.5
        }
    
    def generate_trading_signal(self) -> Optional[Dict]:
        """Generate comprehensive trading signal using all advanced analysis"""
        current_data = self.get_current_market_data()
        if not current_data:
            return None
        
        current_price = current_data['ask']
        
        # Perform complete market analysis
        confluence = self.spatial_temporal_confluence(current_price)
        sr_levels = self.calculate_dynamic_support_resistance()
        trend = self.analyze_multi_timeframe_trend()
        
        # Minimum confluence threshold
        if confluence['confluence_score'] < 0.7:
            return None
        
        # Determine trade direction based on confluence
        if (confluence['trend_analysis']['composite_trend'] == 'BULLISH' and
            sr_levels['price_position'] < 0.5 and  # In lower half of range
            any(zone['strength'] > 0.8 for zone in confluence['time_analysis']['active_time_zones'])):
            
            return self.prepare_buy_signal(current_price, confluence, sr_levels, trend)
        
        elif (confluence['trend_analysis']['composite_trend'] == 'BEARISH' and
              sr_levels['price_position'] > 0.5 and  # In upper half of range
              any(zone['strength'] > 0.8 for zone in confluence['time_analysis']['active_time_zones'])):
            
            return self.prepare_sell_signal(current_price, confluence, sr_levels, trend)
        
        return None
    
    def prepare_buy_signal(self, current_price: float, confluence: Dict, sr_levels: Dict, trend: Dict) -> Dict:
        """Prepare comprehensive buy signal"""
        # Use nearest support as stop loss
        stop_loss = sr_levels['support'][0][1] if sr_levels['support'] else current_price * 0.99
        
        # Calculate take
