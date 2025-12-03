import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ======================
# FIXED OPTIONS SCREENER - AVAILABLE EXPIRIES
# ======================

class RealTimeOptionsScreener:
    def __init__(self, tickers=['SPY', 'QQQ']):
        self.tickers = tickers
        
    def get_current_stock_prices(self):
        """Pobiera AKTUALNE ceny akcji z rynku"""
        current_prices = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                # Pobierz najnowsze dane
                hist = stock.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    hist = stock.history(period='1d')
                    current_price = hist['Close'].iloc[-1]
                
                current_prices[ticker] = current_price
                print(f"   ✅ {ticker}: ${current_price:.2f}")
                
            except Exception as e:
                print(f"   ❌ Błąd pobierania ceny {ticker}: {e}")
                # Aktualne ceny z rynku
                if ticker == 'SPY':
                    current_prices[ticker] = 550.42
                elif ticker == 'QQQ':
                    current_prices[ticker] = 478.15
        return current_prices
    
    def get_available_expiries(self, ticker):
        """Pobiera dostępne daty wygasania dla tickera"""
        try:
            stock = yf.Ticker(ticker)
            expiries = stock.options
            if not expiries:
                # Jeśli brak danych, użyj standardowych expiries
                today = datetime.now()
                expiries = [
                    (today + timedelta(days=30)).strftime('%Y-%m-%d'),
                    (today + timedelta(days=60)).strftime('%Y-%m-%d'),
                    (today + timedelta(days=90)).strftime('%Y-%m-%d')
                ]
            return expiries
        except Exception as e:
            print(f"   ❌ Błąd pobierania expiries dla {ticker}: {e}")
            today = datetime.now()
            return [
                (today + timedelta(days=30)).strftime('%Y-%m-%d'),
                (today + timedelta(days=60)).strftime('%Y-%m-%d')
            ]
    
    def fetch_real_time_options_data(self):
        """Pobiera AKTUALNE dane opcji z dostępnymi expiries"""
        all_options_data = []
        current_prices = self.get_current_stock_prices()
        
        print(f"\n📊 Aktualne ceny akcji:")
        for ticker, price in current_prices.items():
            print(f"   {ticker}: ${price:.2f}")
        
        for ticker in self.tickers:
            print(f"\n🎯 Pobieranie opcji dla {ticker}...")
            
            try:
                stock = yf.Ticker(ticker)
                current_price = current_prices.get(ticker)
                
                # Pobierz dostępne daty wygasania
                available_expiries = self.get_available_expiries(ticker)
                print(f"   📅 Dostępne daty wygasania: {available_expiries[:3]}")  # Pokazuj pierwsze 3
                
                for expiry_str in available_expiries[:3]:  # Analizuj tylko pierwsze 3 daty
                    try:
                        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                        
                        # Pobierz łańcuch opcji
                        opt_chain = stock.option_chain(expiry_str)
                        
                        # Analizuj calls
                        calls_analyzed = 0
                        for _, call in opt_chain.calls.iterrows():
                            if (not pd.isna(call['lastPrice']) and 
                                call['lastPrice'] > 0.10 and
                                call['strike'] > current_price * 0.8 and
                                call['strike'] < current_price * 1.2):
                                
                                option_data = {
                                    'ticker': ticker,
                                    'expiry': expiry_date,
                                    'option_type': 'CALL',
                                    'S': current_price,
                                    'K': call['strike'],
                                    'T': (expiry_date - datetime.now()).days / 365.0,
                                    'lastPrice': call['lastPrice'],
                                    'bid': call['bid'],
                                    'ask': call['ask'],
                                    'volume': call['volume'],
                                    'openInterest': call['openInterest'],
                                    'impliedVolatility': call['impliedVolatility'] if not pd.isna(call['impliedVolatility']) else 0.2,
                                    'moneyness': call['strike'] / current_price
                                }
                                all_options_data.append(option_data)
                                calls_analyzed += 1
                        
                        # Analizuj puts
                        puts_analyzed = 0
                        for _, put in opt_chain.puts.iterrows():
                            if (not pd.isna(put['lastPrice']) and 
                                put['lastPrice'] > 0.10 and
                                put['strike'] > current_price * 0.8 and
                                put['strike'] < current_price * 1.2):
                                
                                option_data = {
                                    'ticker': ticker,
                                    'expiry': expiry_date,
                                    'option_type': 'PUT',
                                    'S': current_price,
                                    'K': put['strike'],
                                    'T': (expiry_date - datetime.now()).days / 365.0,
                                    'lastPrice': put['lastPrice'],
                                    'bid': put['bid'],
                                    'ask': put['ask'],
                                    'volume': put['volume'],
                                    'openInterest': put['openInterest'],
                                    'impliedVolatility': put['impliedVolatility'] if not pd.isna(put['impliedVolatility']) else 0.2,
                                    'moneyness': put['strike'] / current_price
                                }
                                all_options_data.append(option_data)
                                puts_analyzed += 1
                        
                        print(f"   ✅ {expiry_str}: {calls_analyzed} CALLs, {puts_analyzed} PUTs")
                                
                    except Exception as e:
                        print(f"   ❌ Błąd dla {expiry_str}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Błąd dla {ticker}: {e}")
                continue
        
        print(f"\n📊 Łącznie znaleziono {len(all_options_data)} opcji")
        return pd.DataFrame(all_options_data)
    
    def calculate_sophisticated_metrics(self, options_df):
        """Oblicza zaawansowane metryki zysku/ryzyka"""
        metrics_data = []
        
        for _, option in options_df.iterrows():
            try:
                current_price = option['lastPrice']
                spread = option['ask'] - option['bid']
                
                # Wartość wewnętrzna
                if option['option_type'] == 'CALL':
                    intrinsic_value = max(option['S'] - option['K'], 0)
                    time_value = current_price - intrinsic_value
                    breakeven = option['K'] + current_price
                    # Potencjalny zysk (zakładamy 8% ruch)
                    potential_upside = max(option['S'] * 1.08 - option['K'] - current_price, 0)
                else:  # PUT
                    intrinsic_value = max(option['K'] - option['S'], 0)
                    time_value = current_price - intrinsic_value
                    breakeven = option['K'] - current_price
                    # Potencjalny zysk (zakładamy 8% ruch)
                    potential_upside = max(option['K'] - option['S'] * 0.92 - current_price, 0)
                
                # Max strata (cena opcji)
                max_loss = current_price
                
                # Reward/Risk ratio
                reward_risk_ratio = potential_upside / max_loss if max_loss > 0 else 0
                
                # Prawdopodobieństwo zysku (uproszczone Black-Scholes)
                try:
                    d1 = (np.log(option['S'] / option['K']) + (0.05 + 0.5 * option['impliedVolatility']**2) * option['T']) / (option['impliedVolatility'] * np.sqrt(option['T']) + 1e-10)
                    if option['option_type'] == 'CALL':
                        prob_profit = 1 - stats.norm.cdf(d1)
                    else:
                        prob_profit = stats.norm.cdf(d1)
                except:
                    prob_profit = 0.5  # Fallback
                
                # Score płynności
                liquidity_score = np.log1p(option['volume'] or 0) * np.log1p(option['openInterest'] or 0)
                
                # Score zmienności (niższa IV = lepiej)
                iv_score = 1 / (1 + option['impliedVolatility'])
                
                # Łączny score (ważone)
                overall_score = (
                    min(reward_risk_ratio, 3) * 0.4 +
                    prob_profit * 0.3 +
                    iv_score * 0.15 +
                    (1 - min(spread / current_price, 0.5)) * 0.15
                )
                
                metrics = {
                    'ticker': option['ticker'],
                    'expiry': option['expiry'],
                    'option_type': option['option_type'],
                    'strike': option['K'],
                    'current_price': option['S'],
                    'option_price': current_price,
                    'intrinsic_value': intrinsic_value,
                    'time_value': time_value,
                    'breakeven': breakeven,
                    'prob_profit': prob_profit,
                    'reward_risk_ratio': reward_risk_ratio,
                    'potential_upside': potential_upside,
                    'max_loss': max_loss,
                    'spread': spread,
                    'spread_ratio': spread / current_price if current_price > 0 else 0,
                    'implied_vol': option['impliedVolatility'],
                    'liquidity_score': liquidity_score,
                    'days_to_expiry': (option['expiry'] - datetime.now()).days,
                    'overall_score': overall_score
                }
                
                metrics_data.append(metrics)
                
            except Exception as e:
                continue
        
        return pd.DataFrame(metrics_data)
    
    def find_best_options(self):
        """Główna funkcja znajdowania najlepszych opcji"""
        print("🎯 SZUKANIE NAJLEPSZYCH OPCJI...")
        print("=" * 60)
        
        # Pobierz aktualne dane
        options_data = self.fetch_real_time_options_data()
        
        if options_data.empty:
            print("❌ Nie znaleziono opcji! Używam danych testowych...")
            # Fallback: generuj testowe dane
            options_data = self.generate_test_data()
        
        # Oblicz metryki
        metrics_df = self.calculate_sophisticated_metrics(options_data)
        
        if metrics_df.empty:
            print("❌ Nie udało się obliczyć metryk!")
            return None
        
        # Filtry bezpieczeństwa
        filtered_df = metrics_df[
            (metrics_df['option_price'] >= 0.50) &          # Minimum $0.50
            (metrics_df['spread_ratio'] < 0.2) &            # Spread < 20%
            (metrics_df['reward_risk_ratio'] > 0.3) &       # R/R > 0.3
            (metrics_df['prob_profit'] > 0.25) &            # Prob > 25%
            (metrics_df['days_to_expiry'] > 7)              # Min 7 dni
        ].copy()
        
        if filtered_df.empty:
            print("⚠️  Żadna opcja nie spełnia filtrów, używam wszystkich...")
            filtered_df = metrics_df.copy()
        
        # Najlepsze CALLS
        best_calls = filtered_df[
            filtered_df['option_type'] == 'CALL'
        ].nlargest(10, 'overall_score')
        
        # Najlepsze PUTS
        best_puts = filtered_df[
            filtered_df['option_type'] == 'PUT'
        ].nlargest(10, 'overall_score')
        
        return best_calls, best_puts, metrics_df
    
    def generate_test_data(self):
        """Generuje testowe dane jeśli nie ma prawdziwych"""
        print("🔄 Generowanie danych testowych...")
        test_data = []
        current_prices = self.get_current_stock_prices()
        
        for ticker in self.tickers:
            current_price = current_prices.get(ticker, 100)
            # Generuj kilka expiries
            for days in [30, 60, 90]:
                expiry = datetime.now() + timedelta(days=days)
                
                # Generuj calls
                for moneyness in [0.95, 1.0, 1.05]:
                    strike = current_price * moneyness
                    option_price = max(current_price * 0.03 * moneyness, 0.5)
                    
                    test_data.append({
                        'ticker': ticker,
                        'expiry': expiry,
                        'option_type': 'CALL',
                        'S': current_price,
                        'K': strike,
                        'T': days / 365.0,
                        'lastPrice': option_price,
                        'bid': option_price * 0.95,
                        'ask': option_price * 1.05,
                        'volume': 1000,
                        'openInterest': 5000,
                        'impliedVolatility': 0.2,
                        'moneyness': moneyness
                    })
                
                # Generuj puts
                for moneyness in [0.95, 1.0, 1.05]:
                    strike = current_price * moneyness
                    option_price = max(current_price * 0.03 * (2 - moneyness), 0.5)
                    
                    test_data.append({
                        'ticker': ticker,
                        'expiry': expiry,
                        'option_type': 'PUT',
                        'S': current_price,
                        'K': strike,
                        'T': days / 365.0,
                        'lastPrice': option_price,
                        'bid': option_price * 0.95,
                        'ask': option_price * 1.05,
                        'volume': 1000,
                        'openInterest': 5000,
                        'impliedVolatility': 0.22,
                        'moneyness': moneyness
                    })
        
        print(f"✅ Wygenerowano {len(test_data)} testowych opcji")
        return pd.DataFrame(test_data)

# ======================
# SMART RECOMMENDATION ENGINE
# ======================

class SmartOptionsAdvisor:
    def __init__(self):
        self.screener = RealTimeOptionsScreener(['SPY', 'QQQ'])
    
    def generate_recommendations(self):
        """Generuje inteligentne rekomendacje"""
        print("🤖 GENEROWANIE REKOMENDACJI...")
        
        results = self.screener.find_best_options()
        if results is None:
            return None
            
        best_calls, best_puts, all_metrics = results
        
        recommendations = []
        
        # 1. NAJLEPSZY CALL
        if not best_calls.empty:
            best_call = best_calls.iloc[0]
            rec_type = "AGRESYWNY CALL" if best_call['reward_risk_ratio'] > 1.5 else "KONSERWATYWNY CALL"
            
            recommendations.append({
                'action': 'KUP',
                'type': 'CALL',
                'ticker': best_call['ticker'],
                'strike': best_call['strike'],
                'option_price': best_call['option_price'],
                'current_stock_price': best_call['current_price'],
                'expiry': best_call['expiry'].strftime('%Y-%m-%d'),
                'reward_risk': best_call['reward_risk_ratio'],
                'prob_profit': best_call['prob_profit'],
                'potential_return': f"{(best_call['potential_upside'] / best_call['option_price'] * 100):.0f}%",
                'max_loss': f"${best_call['max_loss']:.2f}",
                'reason': f"{rec_type} - R/R: {best_call['reward_risk_ratio']:.2f}, Prob: {best_call['prob_profit']:.1%}"
            })
        
        # 2. NAJLEPSZY PUT
        if not best_puts.empty:
            best_put = best_puts.iloc[0]
            rec_type = "ZABEZPIECZENIE" if best_put['prob_profit'] > 0.6 else "SPEKULACJA"
            
            recommendations.append({
                'action': 'KUP', 
                'type': 'PUT',
                'ticker': best_put['ticker'],
                'strike': best_put['strike'],
                'option_price': best_put['option_price'],
                'current_stock_price': best_put['current_price'],
                'expiry': best_put['expiry'].strftime('%Y-%m-%d'),
                'reward_risk': best_put['reward_risk_ratio'],
                'prob_profit': best_put['prob_profit'],
                'potential_return': f"{(best_put['potential_upside'] / best_put['option_price'] * 100):.0f}%",
                'max_loss': f"${best_put['max_loss']:.2f}",
                'reason': f"PUT {rec_type} - R/R: {best_put['reward_risk_ratio']:.2f}, Prob: {best_put['prob_profit']:.1%}"
            })
        
        return recommendations, len(all_metrics)

# ======================
# MAIN EXECUTION
# ======================

def main():
    print("🎯 SMART OPTIONS SCREENER - REAL TIME")
    print("📅 Opcje z dostępnymi datami wygasania")
    print("💰 Aktualne ceny z rynku: SPY i QQQ")
    print("=" * 70)
    
    advisor = SmartOptionsAdvisor()
    results = advisor.generate_recommendations()
    
    if results is None:
        print("❌ Nie udało się wygenerować rekomendacji!")
        return
    
    recommendations, total_analyzed = results
    
    # Wyświetl rekomendacje
    print("\n" + "=" * 70)
    print("🎯 REKOMENDOWANE OPCJE DO KUPNA")
    print("=" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. ✅ {rec['action']} {rec['type']} {rec['ticker']}")
        print(f"   ├── Strike: ${rec['strike']:.2f}")
        print(f"   ├── Cena akcji: ${rec['current_stock_price']:.2f}")
        print(f"   ├── Cena opcji: ${rec['option_price']:.2f}")
        print(f"   ├── Wygasanie: {rec['expiry']}")
        print(f"   ├── Reward/Risk: {rec['reward_risk']:.2f}")
        print(f"   ├── Prawd. zysku: {rec['prob_profit']:.1%}")
        print(f"   ├── Potencjalny zysk: {rec['potential_return']}")
        print(f"   ├── Maks. strata: {rec['max_loss']}")
        print(f"   └── 📝 {rec['reason']}")
    
    # Podsumowanie
    print(f"\n📊 PODSUMOWANIE ANALIZY:")
    print(f"   Przeanalizowanych opcji: {total_analyzed}")
    print(f"   Zalecane pozycje: {len(recommendations)}")
    print(f"   Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Ostrzeżenie
    print(f"\n⚠️  WAŻNE INFORMACJE:")
    print(f"   • Program używa dostępnych dat wygasania z Yahoo Finance")
    print(f"   • Sprawdź aktualne ceny przed zawarciem transakcji")
    print(f"   • Inwestowanie w opcje wiąże się z ryzykiem straty")

if __name__ == "__main__":
    main()