# 📈 Physics-Informed Neural Networks for Portfolio Optimization

This project explores the use of **Physics-Informed Neural Networks (PINNs)** and deep learning for solving **dynamic portfolio optimization** problems. We combine methods from reinforcement learning, financial econometrics, and partial differential equations (PDEs) to optimize investment strategies under uncertainty.

## 🧠 Core Idea

We solve the **Hamilton-Jacobi-Bellman (HJB)** equation using PINNs to learn the **optimal value function and policy**. Additionally, we use **LSTM** networks to forecast asset returns and integrate them into a real-time **dynamic allocation** strategy. Benchmarking is done against traditional econometric models (**ARIMA**, **GARCH**) and fixed strategies.

---

## 📂 Project Structure

- `ENM_5320_Final_Project.ipynb` – Implementation notebook containing model training, simulations, and visualizations.
- `PINN for Portfolio Optimization_Final Report.pdf` – Comprehensive report covering theoretical background, methods, results, and analysis.

---

## 🧩 Methods Used

### 1. PINNs for Solving HJB PDE
- Input: time `t`, wealth `w`
- Output: Value function `V(t, w)`
- Loss: HJB residual + terminal utility condition
- Framework: JAX

### 2. FDM (Finite Difference Method)
- Backward Euler time-stepping
- Grid-based numerical solver used as a baseline

### 3. LSTM for Return Forecasting
- Input: 20-day rolling return sequence
- Output: Next-day return
- Framework: TensorFlow

### 4. ARIMA and GARCH Models
- ARIMA(1,0,1) for return autocorrelation
- GARCH(1,1) for volatility clustering
- Framework: statsmodels

---

## 📊 Dataset

- Assets: `MSFT`, `AAPL`, `GOOG`, `AMZN`, `SPY`
- Source: Yahoo Finance (2020–2025)
- Daily log returns used
- Preprocessing: Forward-filling, ADF test for stationarity

---

## 📈 Evaluation Metrics

| Strategy     | Final Wealth | Sharpe Ratio | Max Drawdown | Volatility |
|--------------|--------------|---------------|----------------|-------------|
| PINN + LSTM  | $220,000     | 0.90          | 40%            | 30%         |
| SPY Benchmark| $160,000     | 0.65          | 30%            | 18%         |
| Fixed (60%)  | $180,000     | 0.75          | 35%            | 22%         |

---

## 🔍 Key Insights

- **PINN solutions** align closely with analytical results from the Merton problem.
- **LSTM-guided strategies** outperform static benchmarks under realistic constraints.
- **FDM** provides high accuracy but lacks flexibility.
- **Traditional models** (ARIMA/GARCH) offer supporting insights but fall short on dynamic performance.

---

## ⚠️ Challenges

- Balancing PDE losses in PINNs
- Overfitting in LSTM due to limited financial data
- FDM grid sensitivity and boundary conditions
- Lag in integrating model outputs into portfolio rebalancing

---

## 🚀 Future Directions

- Add **transaction costs, slippage, and liquidity effects**
- Use **Transformer-based** return predictors
- Combine **GARCH volatility modeling** with PINNs
- Deploy in **robo-advisors** or **automated trading systems**

---

## 📚 References

- Merton, R. (1971). *Optimum consumption and portfolio rules...*
- Raissi, M. et al. (2019). *Physics-Informed Neural Networks...*
- Engle, R. (1982). *ARCH modeling in econometrics...*
- Tsay, R. (2010). *Analysis of Financial Time Series*

---

## 💼 Authors

This project was developed as part of the **ENM 5320: Final Project** at the University of Pennsylvania.

---

