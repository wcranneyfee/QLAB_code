{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "moment of intertia = 5.78e-05 +/- 1.63e-06 kg m^2\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from SDOM_analysis import SDOM_analysis\n",
    "\n",
    "\n",
    "def rpm_to_w(rpm):\n",
    "    f = (rpm*2*math.pi)/60;\n",
    "    return f\n",
    "    \n",
    "cue_ball_mass = 0.178; # kg\n",
    "dM = 0.001; # kg\n",
    "\n",
    "ball_r = 2.85e-2; # meters\n",
    "dRball = 0.001; # meters\n",
    "\n",
    "\n",
    "Current = np.array([1.998, 1.799, 1.599, 1.399, 1.198, 0.999]); # A\n",
    "\n",
    "dT = 0.1; # A\n",
    "\n",
    "N = 200 # no units\n",
    "mu_0 = 4e-7*math.pi # H/m\n",
    "D = 21e-2 # [m], diameter of helmholtz coils\n",
    "\n",
    "I_s = (2/5) * cue_ball_mass * (ball_r) ** 2\n",
    "\n",
    "d_r_squared = (2 * dRball)/(ball_r) * ball_r**2\n",
    "\n",
    "dI_s = (2/5) * math.sqrt((d_r_squared/ball_r**2)**2 + (dM/cue_ball_mass)**2) * I_s\n",
    "\n",
    "\n",
    "print(f\"moment of intertia = {I_s:.2e} +/- {dI_s:.2e} kg m^2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_osc = pd.read_csv(\"Data/lab4_osc(in).csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Trial</th>\n",
       "      <th>B-Direction</th>\n",
       "      <th>T_o(s)</th>\n",
       "      <th>dT</th>\n",
       "      <th>I(Amps)</th>\n",
       "      <th>dI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Up</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1.999</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Up</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1.799</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Up</td>\n",
       "      <td>1.8</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1.599</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Up</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1.399</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Up</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1.199</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>Up</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.998</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Trial  B-Direction  T_o(s)   dT  I(Amps)     dI\n",
       "0       1          Up     1.5  0.1    1.999  0.005\n",
       "1       2          Up     1.7  0.1    1.799  0.005\n",
       "2       3          Up     1.8  0.1    1.599  0.005\n",
       "3       4          Up     2.0  0.1    1.399  0.005\n",
       "4       5          Up     2.3  0.1    1.199  0.005\n",
       "5       6          Up     2.4  0.1    0.998  0.005"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_osc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "B = (16*N*mu_0*Current)/(math.sqrt(125)*D) # T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.00342201, 0.00308118, 0.00273863, 0.00239609, 0.00205183,\n",
       "       0.001711  ])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "B"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "T = np.array(df_osc['T_o(s)'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.5, 1.7, 1.8, 2. , 2.3, 2.4])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "mu_osc = (4 * (math.pi**2) * I_s)/(T**2 *B)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.29652814, 0.25639816, 0.25730631, 0.23821341, 0.21034463,\n",
       "       0.23166261])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mu_osc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mean = 2.484089e-01\n",
      "SDOM = 1.196067e-02\n",
      "STDEV = 2.929754e-02\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(0.24840887403060355, 0.029297537207731456, 0.011960669479857787)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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JSVhx44dWLJkCXbt2qU33tHRUetnz1ZWVnB2dka3bt10xqelpWndO8/Ozg52dnb15qRWqwEApaWl4rkqZ8+eRWZmJpYvXy7G+fn5YcOGDfD19UV4eDgGDx6M11577aHL3FyxYCEiolbp2rVrGDt2LBYsWICwsDB4eXlh0KBByM7Oho+PzyP3X1hYiFOnTuHzzz+XNJ+fnx+srKwwf/58LFq0CFeuXMHs2bMRFRWldVX5JUuWID4+HkOHDoWtrW2LLlYAnsNCRESt0G+//YYxY8YgJCQEb731FgDAx8cHwcHBWLRoUaO8xxdffAE/Pz84OztLmq99+/bYuXMnTp48iX79+uH1119HVFQU1qxZoxU3btw4/PTTT0hPT8fatWsbJWc54wgLERG1Oo6OjsjNza3TnpaWJqmfwzV/vqSjr5CQEKmpAdAUI+PGjas35uTJk7h79y66d+8OU13HzloYjrAQERE1gYCAAEyZMqVJ+s7Ly8OMGTNw6NAhXLp0SWfx1dKwYCEiImoC8+fPh7u7e6P3W1paikmTJmH9+vXw9PTE/Pnz8fbbbzf6+8hNyx9DIiIiakGsrKxw/PhxcXrKlClNNpIjJxxhISIiItljwUJERESyx4KFiIiIZI8FCxEREckeCxYiIiKSPRYsREREJHssWIiIiEj2WLAQERGR7LFgISIiItljwUJERK1afHw8bG1txUd8fLyxUyIdeGl+IiJq1aKiojB58mRx2tHRsdHfIyIiAh06dMA777zT6H23FixYiIioVXN0dGxwkTJ8+HBEREQgIiJCb4xarcaXX36JvXv3NjBDAnhIiIiIWrGUlBRYWloiLy9PbJsxYwb69euHoqKiRnmPb775BiYmJvDz85M034kTJ/DMM8/AyckJCoVC63H37t1Gya05YcFCRESt1ksvvYQePXogISEBABAXF4f09HTs378fDg4OjfIee/fuRXBwMExMDP/K/f777zF8+HD0798fmZmZOHDgABwdHTFixAikpqaiTZs2jZJbc8JDQkRE1DQqVPpfUygBpaVhsTABTK0eHmtqIyk9AFAoFFixYgUmTZqEjh07Yu3atcjKykKnTp0k96XP3r17sWrVKknzvP766xg/fjzef/99AICXlxemTJmCb7/9Vut8m9aEBQsRETWNnbb6X+v4HDD8y+rpz5yByvu6Y50DgZGHq6fT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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "SDOM_analysis(len(mu_osc), mu_osc, mu_osc/10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_prec = pd.read_csv(\"Data/lab4_prec(in).csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Trial</th>\n",
       "      <th>B-Direction</th>\n",
       "      <th>T_p(s)</th>\n",
       "      <th>dT</th>\n",
       "      <th>w_s(RPM)</th>\n",
       "      <th>dw_s</th>\n",
       "      <th>I(Amps)</th>\n",
       "      <th>dI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Up</td>\n",
       "      <td>7.6</td>\n",
       "      <td>0.2</td>\n",
       "      <td>220</td>\n",
       "      <td>10</td>\n",
       "      <td>1.998</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Up</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>210</td>\n",
       "      <td>10</td>\n",
       "      <td>1.799</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Up</td>\n",
       "      <td>11.8</td>\n",
       "      <td>0.2</td>\n",
       "      <td>240</td>\n",
       "      <td>10</td>\n",
       "      <td>1.599</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Up</td>\n",
       "      <td>12.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>235</td>\n",
       "      <td>10</td>\n",
       "      <td>1.399</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Up</td>\n",
       "      <td>13.2</td>\n",
       "      <td>0.2</td>\n",
       "      <td>205</td>\n",
       "      <td>10</td>\n",
       "      <td>1.198</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>Up</td>\n",
       "      <td>18.9</td>\n",
       "      <td>0.2</td>\n",
       "      <td>216</td>\n",
       "      <td>10</td>\n",
       "      <td>0.999</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>Down</td>\n",
       "      <td>6.0</td>\n",
       "      <td>0.2</td>\n",
       "      <td>230</td>\n",
       "      <td>10</td>\n",
       "      <td>1.998</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2</td>\n",
       "      <td>Down</td>\n",
       "      <td>5.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>180</td>\n",
       "      <td>10</td>\n",
       "      <td>1.799</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3</td>\n",
       "      <td>Down</td>\n",
       "      <td>5.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>135</td>\n",
       "      <td>10</td>\n",
       "      <td>1.599</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>4</td>\n",
       "      <td>Down</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.2</td>\n",
       "      <td>130</td>\n",
       "      <td>10</td>\n",
       "      <td>1.399</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>5</td>\n",
       "      <td>Down</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.2</td>\n",
       "      <td>165</td>\n",
       "      <td>10</td>\n",
       "      <td>1.199</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>6</td>\n",
       "      <td>Down</td>\n",
       "      <td>7.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>150</td>\n",
       "      <td>10</td>\n",
       "      <td>0.998</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Trial  B-Direction  T_p(s)   dT  w_s(RPM)  dw_s  I(Amps)     dI\n",
       "0        1          Up     7.6  0.2       220    10    1.998  0.005\n",
       "1        2          Up     7.5  0.2       210    10    1.799  0.005\n",
       "2        3          Up    11.8  0.2       240    10    1.599  0.005\n",
       "3        4          Up    12.4  0.2       235    10    1.399  0.005\n",
       "4        5          Up    13.2  0.2       205    10    1.198  0.005\n",
       "5        6          Up    18.9  0.2       216    10    0.999  0.005\n",
       "6        1        Down     6.0  0.2       230    10    1.998  0.005\n",
       "7        2        Down     5.5  0.2       180    10    1.799  0.005\n",
       "8        3        Down     5.5  0.2       135    10    1.599  0.005\n",
       "9        4        Down     5.0  0.2       130    10    1.399  0.005\n",
       "10       5        Down     7.0  0.2       165    10    1.199  0.005\n",
       "11       6        Down     7.4  0.2       150    10    0.998  0.005"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_prec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_up = df_prec.loc[df_prec['B-Direction'] == 'Up']\n",
    "df_down = df_prec.loc[df_prec['B-Direction'] == 'Down']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Trial</th>\n",
       "      <th>B-Direction</th>\n",
       "      <th>T_p(s)</th>\n",
       "      <th>dT</th>\n",
       "      <th>w_s(RPM)</th>\n",
       "      <th>dw_s</th>\n",
       "      <th>I(Amps)</th>\n",
       "      <th>dI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Up</td>\n",
       "      <td>7.6</td>\n",
       "      <td>0.2</td>\n",
       "      <td>220</td>\n",
       "      <td>10</td>\n",
       "      <td>1.998</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Up</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>210</td>\n",
       "      <td>10</td>\n",
       "      <td>1.799</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Up</td>\n",
       "      <td>11.8</td>\n",
       "      <td>0.2</td>\n",
       "      <td>240</td>\n",
       "      <td>10</td>\n",
       "      <td>1.599</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Up</td>\n",
       "      <td>12.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>235</td>\n",
       "      <td>10</td>\n",
       "      <td>1.399</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Up</td>\n",
       "      <td>13.2</td>\n",
       "      <td>0.2</td>\n",
       "      <td>205</td>\n",
       "      <td>10</td>\n",
       "      <td>1.198</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>Up</td>\n",
       "      <td>18.9</td>\n",
       "      <td>0.2</td>\n",
       "      <td>216</td>\n",
       "      <td>10</td>\n",
       "      <td>0.999</td>\n",
       "      <td>0.005</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Trial  B-Direction  T_p(s)   dT  w_s(RPM)  dw_s  I(Amps)     dI\n",
       "0       1          Up     7.6  0.2       220    10    1.998  0.005\n",
       "1       2          Up     7.5  0.2       210    10    1.799  0.005\n",
       "2       3          Up    11.8  0.2       240    10    1.599  0.005\n",
       "3       4          Up    12.4  0.2       235    10    1.399  0.005\n",
       "4       5          Up    13.2  0.2       205    10    1.198  0.005\n",
       "5       6          Up    18.9  0.2       216    10    0.999  0.005"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "w_s = np.array(df_prec['w_s(RPM)'])\n",
    "w_s = rpm_to_w(w_s)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([23.03834613, 21.99114858, 25.13274123, 24.60914245, 21.4675498 ,\n",
       "       22.61946711, 24.08554368, 18.84955592, 14.13716694, 13.61356817,\n",
       "       17.27875959, 15.70796327])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "w_s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "w_p = np.array(df_prec['T_p(s)'])\n",
    "w_p = (2*math.pi)/w_p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.82673491, 0.83775804, 0.53247333, 0.50670849, 0.47599889,\n",
       "       0.33244367, 1.04719755, 1.14239733, 1.14239733, 1.25663706,\n",
       "       0.8975979 , 0.8490791 ])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "w_p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.00342201, 0.00308118, 0.00273863, 0.00239609, 0.00205183,\n",
       "       0.001711  , 0.00342201, 0.00308118, 0.00273863, 0.00239609,\n",
       "       0.00205183, 0.001711  ])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "B_prec = np.concatenate((B, B))\n",
    "B_prec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.3218890947651393, 0.3457956509508091, 0.2826008277239735, 0.3009685538514567, 0.28801544035424226, 0.2541669735288373] [0.40417673487756917, 0.3410478170941589, 0.41290324153918995, 0.4371412049418394, 0.45080290913053905]\n"
     ]
    }
   ],
   "source": [
    "mu_prec = (I_s * w_s * w_p)/(B_prec)\n",
    "\n",
    "mu_prec_up = [n for i, n in enumerate(mu_prec) if i < len(mu_prec)/2]\n",
    "mu_prec_down = [n for i, n in enumerate(mu_prec) if i > len(mu_prec)/2]\n",
    "print(mu_prec_up, mu_prec_down)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mean = 3.554806e-01\n",
      "SDOM = 1.964893e-02\n",
      "STDEV = 6.806587e-02\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(0.35548063699556187, 0.06806587424841838, 0.019648925409975787)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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ra1w+PBhg69dVlYmKJVKYefOnTrlc+fOFYYMGVKnPgBh165dBturYsm/ESKyXPHx8QIAk5f4+HipQ25wxv5+12T2OCxyZco4LHXG2FCrmzZIR8d6Hfbll19i0qRJeO2117BixQocOXLEYO/H77//jt9//91oex06dNAZj+bs2bMICwvDvXv34OTkhE8//RSPP/64weP37duHu3fvolu3brhx4wbeeOMN/PTTTzh37hzatGljUh21Wg13d3d8++23CAsLE9ueMWMG7t69i08//VTva//666/o0KEDvv/+e51LhklJSdi0aRMuXLigU1+hUGDXrl2YMGGC0c/E4G+EiKgR1e5hKS0tRXh4OAAgIyOjzthhzbGHxdRxWFrMXEJ6OTk17evVMzccO3YsAgMDkZiYiIMHDxq9VOPm5mb2PS7du3dHdnY2bt++jR07diAmJgbfffcdAgMD9davebnl4YcfRlhYGLp06YJNmzZhwYIFJtXJycnBvXv3MHLkSJ22y8vL0bdvXwDAJ598gtjYWHHfvn370KVLFwDmP1VGRCRHtRMQdY3/SAcFBcGxnv/RbY5adsJiIQ4cOICffvpJ7xg4tSUlJd33sed9+/bpzP1kY2ODrl27AgBCQkJw6tQprFy5EikpKSbF5+joiIcffljnya/71amahPKrr74S782pYmtrCwCIiorSmZqhQ4cOUCqVDfKkGhERWZaWnbCUlEgdwX2dOXMGTz31FFJSUrB161YsW7YMn332mcH6cXFxmDx5stE2aycItQmCgLKyMpNjLCsrw/nz5w1OgKmvTmBgIGxtbXHt2jVEREToPcbZ2RnOzs51yqueVKs5OWZaWhrGjx9vcsxERGRZWnbCIvOutqtXr2LMmDFYtGgRpk6disDAQPTv3x+ZmZkIDg7We4y5l4ReffVVjB49Gj4+Prhz5w62bt2Kw4cPY//+/WKd1atXY9euXfjmm28AAC+//DLGjRuHjh07orCwEG+88QaKi4sRExMjHnO/Os7Oznj55Zfx0ksvQaPRIDw8HMXFxTh69CicnJx02qrtfk+qlZSU4NKlS2L9K1euIDs7G25ubkYn6CQiIhlrghuAm0Rze0ro1q1bQo8ePYQXX3xRpzwqKkoYNWpUg73OCy+8ID7x1bZtW2H48OHCwYMHderEx8cLvr6+4nZ0dLTg5eUltGrVSmjf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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "SDOM_analysis(len(mu_prec), mu_prec, mu_prec/10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mean = 2.736575e-01\n",
      "SDOM = 1.136808e-02\n",
      "STDEV = 3.938019e-02\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(0.2736574821131733, 0.03938019173856555, 0.01136808215049995)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjUAAAGwCAYAAABRgJRuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABa/0lEQVR4nO3de1RU5foH8O8wyIDcFBEQuYi3FO+CchFDT4qZdzORjoqlKziiaaxO6TETzMDMLPMWlD8vp0TtpGllIZUGpKYhdFRIsTSEQNKOICOCMPv3x8SGYWZgBoEZhu9nrVlr9jvPvPuZ2ePm8d17v1siCIIAIiIiojbOzNAJEBERETUHFjVERERkEljUEBERkUlgUUNEREQmgUUNERERmQQWNURERGQSWNQQERGRSTA3dAKtSaFQ4Pfff4etrS0kEomh0yEiIiIdCIKAu3fvwtXVFWZm2sdj2lVR8/vvv8Pd3d3QaRAREVET3LhxA25ublpfb1dFja2tLQDll2JnZ2fgbIiIiEgXpaWlcHd3F/+Oa9OuipqaQ052dnYsaoiIiNqYxk4d4YnCREREZBJY1BAREZFJYFFDREREJqFdnVNDRESGV11djQcPHhg6DTIiHTp0gFQqfeh+WNQQEVGrEAQBRUVFuHPnjqFTISPUqVMnuLi4PNQ8cixqiIioVdQUNE5OTujYsSMnQSUAymL33r17KC4uBgB069atyX2xqCEiohZXXV0tFjRdunQxdDpkZKysrAAAxcXFcHJyavKhKJ4oTERELa7mHJqOHTsaOBMyVjW/jYc534pFDRERtRoeciJtmuO3waKGiIiITAKLGiIiIh3ExcXBxsZGfMTFxRk6JaqHRQ0REZEOIiMjkZWVJT4iIyObre/4+HiMGDECtra2cHJywvTp03H58mWt8T169IBEIlF7REVFae1fIpFg+fLlzZLv9u3b4eXlBUtLS/j4+CAtLU3l9dTUVEyZMgWurq6QSCT49NNPm2W9jWFRQ0REpAMHBwf07t1bfDg4ODRb39999x2ioqJw5swZpKSkoKqqCiEhIZDL5Rrjz507h8LCQvGRkpICAHjqqac0xiYmJmLw4MHNkuuBAwewfPlyrFq1CpmZmRg9ejQmTpyIvLw8MUYul2PIkCHYunVrs6xTV7yk20TU/LB11a1bt4eaC4CIqD1JSkrCM888g19++QXdu3cHACxatAhnz55FWloa7O3tH6r/r776SmV5165dcHJyQkZGBh599FG1+K5du6osr1+/Hr169UJwcLBKe1lZGf7+97/j/fffx7p169T6EQQBb775Jt577z0UFhaib9++WL16NWbNmqU1102bNmHhwoVYtGgRAOCdd95BcnIyduzYgfj4eADAxIkTMXHiRN0+fDPiSI2JSEhIgI+Pj86PhIQEQ6dMRNRmzJkzB4888oj4Rzs2NhbJycn48ssvNRY09c+/0fSof8imrpKSEgDQaTSosrISH374IZ599lm1K4iioqIwadIkjBs3TuN7X3nlFezatQs7duzApUuX8MILL2Du3Ln47rvvtK4rIyMDISEhKu0hISE4depUo7m2NI7UmIiIiAhMnTpVXC4vL0dQUBAAID09XZzYqAZHaYjIaGg5xNJirK31fotEIsHrr7+OWbNmwdXVFZs3b0ZaWpo4alNfZGQkZs+e3WCf2t4rCAKio6MRFBSEgQMHNprbp59+ijt37mDBggUq7fv378f58+dx7tw5je+Ty+XYtGkTvv32WwQEBAAAevbsifT0dCQkJKiN+gDArVu3UF1dDWdnZ5V2Z2dnFBUVNZprS2NRYyLqH06qexx26NChsG7CP2IiolZhY9O66xOEJr1t8uTJ8Pb2RmxsLI4fP44BAwZojXVwcGjyOTdLlizBf//7X6Snp+sUv3PnTkycOBGurq5i240bN7Bs2TIcP34clpaWGt+XnZ2N+/fvY/z48SrtlZWVGDZsGD766CNERESI7V9++SV69eoFQH1OGUEQjGIOIhY1REREOkhOTsbPP/+scaSivri4uEYv+f7yyy8xevRolbalS5fi6NGjSE1NhZubW6M5/fbbb/j6669x6NAhlfaMjAwUFxfDx8dHbKuurkZqaiq2bt2KiooKKBQKAMAXX3yhNmokk8nQqVMn+Pn5iW3du3eHVCqFVCpVG5UpLi5u9DtpDSxqiIjIsMrKDJ1Bo86fP4+nnnoKCQkJ2L9/P1avXo2PP/5Ya7y+h58EQcDSpUtx+PBhnDx5El5eXjrlVXNC8aRJk1TaH3vsMVy4cEGl7ZlnnkG/fv3w8ssvQyqVwtvbGzKZDHl5eRoPNQGAra2tWpuPjw9SUlIwY8YMsS0lJQXTpk3TKeeWxKKGiIgMy8gPj1+/fh2TJk3CihUrMG/ePHh7e2PEiBHIyMhQGQmpS9/DT1FRUdi3bx+OHDkCW1tbcSTE3t5ePCdy69atOHz4ML755hsAgEKhwK5duxAeHg5zc9U/57a2tmrn41hbW6NLly5iu62tLV588UW88MILUCgUCAoKQmlpKU6dOgUbGxuEh4drzDU6Ohrz5s2Dr68vAgICkJiYiLy8PJV5e8rKynD16lVx+dq1a8jKyoKDgwM8PDx0/l70JjTBtm3bhB49eggymUwYPny4kJqaqjU2LS1NCAwMFBwcHARLS0vhkUceETZt2qQSExwcLABQezzxxBNizJo1a9Red3Z21ivvkpISAYBQUlKi3wdug8rKysTvqayszNDpEFE7V15eLmRnZwvl5eWGTkUvt2/fFvr16yc899xzKu1Tp04VJkyY0Gzr0fQ3EICwa9cuMWbNmjWCp6enuJycnCwAEC5fvqzTOoKDg4Vly5aptCkUCmHz5s3CI488InTo0EHo2rWrMGHCBOG7775rsK9t27YJnp6egoWFhTB8+HC1+BMnTmj8POHh4Vr7bOg3ouvfb4kg6HfG1IEDBzBv3jxs374do0aNQkJCAj744ANkZ2drrL4yMzPx888/Y/DgwbC2tkZ6ejoiIiLw9ttv47nnngMA/Pnnn6isrBTfc/v2bQwZMgQffPCBeDZ3TEwM/vOf/+Drr78W46RSqdq1+g0pLS2Fvb09SkpKYGdnp8/HbnPkcjls/jr5rqysjCcKE5FB3b9/H9euXRNnoSWqr6HfiK5/v/U+/KTLpDt1DRs2DMOGDROXe/TogUOHDiEtLU0sauoP0e3fvx8dO3ZUmxnR3NwcLi4uOudaUVGBiooKcbm0tFTn9xIREVHbotfke80x6U5mZiZOnTql9aQkQHl52pw5c9RGF3Jzc+Hq6govLy/MmTMHv/76a4Prio+Ph729vfhwd3fXKUciIiJqe/Qqah5m0h03NzfIZDL4+voiKipKHOmp7+zZs7h48aLa635+fti7dy+Sk5Px/vvvo6ioCIGBgbh9+7bWda5cuRIlJSXi48aNGzp+UiIiImprmnT1U1Mm3UlLS0NZWRnOnDmDFStWoHfv3ggLC1OL27lzJwYOHIiRI0eqtNe9h8SgQYMQEBCAXr16Yc+ePYiOjta4TplMBplMpuvHIiIiojZMr6LG0dGxyZPu1FxzP2jQINy8eRMxMTFqRc29e/ewf/9+rF27ttFcrK2tMWjQIOTm5urzEYiIiMhE6XX4ycLCQpx0p66UlBQEBgbq3I8gCCon8NY4ePAgKioqMHfu3Eb7qKioQE5ODu9hRERERACacPipsUl3Vq5ciYKCAuzduxcAsG3bNnh4eKBfv34AlDdX3LhxI5YuXarW986dOzF9+nR06dJF7bUXX3wRU6ZMgYeHB4qLi7Fu3TqUlpZqnRyIiIiI2he9i5rQ0FDcvn0ba9euRWFhIQYOHIhjx47B09MTAFBYWIi8vDwxXqFQYOXKlbh27RrMzc3Rq1cvrF+/XuUmWQBw5coVpKen4/jx4xrXm5+fj7CwMNy6dQtdu3aFv78/zpw5I66XiIiI2je9J99ryzj5HhGRYXDyPWpMc0y+p9c5NURERETGikUNERERmQQWNURERDqIi4uDjY2N+IiLizN0SlRPkybfIyIiam8iIyMxe/Zscbn+fQubw4IFC+Di4oL169c3e9/tAYsaIiIiHTg4ODS5kBkzZgwWLFiABQsWaI1RKBT44osvcPTo0SZmSDz8RERE1IikpCRYWlqioKBAbFu0aBEGDx6MkpKSZlnH999/DzMzM/j5+en1vjNnzuCxxx6Do6MjJBKJyuPOnTvNkltbwaKGiIgMSi7X/1FVVfv+qiplW3m5bv02xZw5c/DII48gPj4eABAbG4vk5GR8+eWXsLe3b+InV3X06FFMmTIFZma6/2n+6aefMGbMGAwZMgSpqan46quv4ODggLFjx+LAgQPo1KlTs+TWVvDwExERGdRfU2rp5eBB4KmnlM8PHwZmzwaCg4GTJ2tjevQAbt1Sf29TZmeTSCR4/fXXMWvWLLi6umLz5s1IS0tD9+7d9e9Mi6NHj2Ljxo16vef555/HtGnTsGnTJgCAt7c3wsLC8MMPP6ic/9NecKSGiIhIB5MnT4a3tzdiY2Nx+PBhDBgwQGts/Sul0tLSEBkZqdZWIycnB/n5+Rg3bhwA4KOPPtIaW+PmzZtIT0/H4sWLVdqtra0hkUhU2mbNmoWhQ4di6NChD/ENGD+O1BARkUGVlen/Hpms9vmMGco+6h+1uX79odJSk5ycjJ9//hnV1dVwdnZuMLb+lVJ///vf8eSTT2LmzJliW91RnqNHj2L8+PGwsrICAEydOlXl3BpNI0IZGRlQKBQYMmSIWruvr69K23/+8x8dPmHbx6KGiIgM6mHv4mJurnw0d791nT9/Hk899RQSEhKwf/9+rF69Gh9//LHW+PpXSllZWcHJyQm9e/fWGH/kyBEsWrRIXLa1tYWtrW2DOSkUCgBAeXm5eO7MhQsXkJqairVr14pxfn5+2LZtG3x9fREeHg5/f3/84x//aPQzt0UsaoiIiBpw/fp1TJo0CStWrMC8efPg7e2NESNGICMjAz4+Pg/df3FxMc6dO4dPP/1Ur/f5+fnBysoKL730ElatWoVffvkFS5cuRWRkJAIDA8W41atXIy4uDqNGjYKNjY3JFjQAz6khIiLS6s8//8TEiRMxdepU/Otf/wIA+Pj4YMqUKVi1alWzrOOzzz6Dn58fnJyc9Hpf165dcfDgQZw9exaDBw/G888/j8jISLzzzjsqcZMnT8avv/6K5ORkbN68uVlyNlYcqSEiItLCwcEBOTk5au1HjhzRq5+TdS/L0tDX1KlT9U0NgLJgmTx5coMxZ8+exZ07d9C3b1+YazpOZ0I4UkNERGRAQUFBCAsLa5G+CwoKsGjRIpw4cQK5ubkaCzRTwqKGiIjIgF566SW4u7s3e7/l5eWYNWsWtm7dCi8vL7z00ktYt25ds6/HmJj2OBQREVE7ZWVlhdOnT4vLYWFhLTYiZCw4UkNEREQmgUUNERERmQQWNURERGQSWNQQERGRSWBRQ0RERCaBRQ0RERGZBBY1REREZBJY1BAREZFJYFFDREREJoFFDRERkQ7i4uJgY2MjPuLi4gydEtXD2yQQERHpIDIyErNnzxaXHRwcmn0dCxYsgIuLC9avX9/sfbcHLGr0VFhYiMLCQp3ju3Xrhm7durVgRkRE1BocHByaXMiMGTMGCxYswIIFC7TGKBQKfPHFFzh69GgTMyQeftJTQkICfHx8dH4kJCQYOmUiInpISUlJsLS0REFBgdi2aNEiDB48GCUlJc2yju+//x5mZmbw8/PT631nzpzBY489BkdHR0gkEpXHnTt3miW3toIjNXqKiIjA1KlTxeXy8nIEBQUBANLT02FlZaUSz1EaIqK2b86cOVi/fj3i4+OxdetWxMbGIjk5GWfOnIG9vX2zrOPo0aOYMmUKzMx0H2/46aefMGbMGCxevBhbtmzBjRs38PTTT2PIkCGIjIxEp06dmiW3toJFjZ7qH06Sy+Xi86FDh8La2toQaRERtV1Vcu2vSaSA1FK3WJgB5nX+Y6kt1lz//bREIsHrr7+OWbNmwdXVFZs3b0ZaWhq6d++ud1/aHD16FBs3btTrPc8//zymTZuGTZs2AQC8vb0RFhaGH374QeX8n/aCRQ0RERnWQRvtr7k+AYz5onb5Eyeg+p7mWKdgYNzJ2uUjPYCKW+pxTwtNyRKTJ0+Gt7c3YmNjcfz4cQwYMEBrbFxcnMrVUeXl5Thz5gyWLFkitn355ZcYPXo0ACAnJwf5+fkYN24cAOCjjz5CRESExtgaN2/eRHp6Or799luVdmtra0gkkiZ9xrauSefUbN++HV5eXrC0tISPjw/S0tK0xqanp2PUqFHo0qULrKys0K9fP7z99tsqMbt371Y7DiiRSHD//v0mr5eIiKg5JScn4+eff0Z1dTWcnZ0bjI2MjERWVpb48PX1xdq1a9Xaahw9ehTjx48XT2GYOnWq1tgaGRkZUCgUGDJkiFq7pvj2QO+RmgMHDmD58uXYvn07Ro0ahYSEBEycOBHZ2dnw8PBQi7e2tsaSJUswePBgWFtbIz09HREREbC2tsZzzz0nxtnZ2eHy5csq77W0rB1y1He9RETURswu0/6aRKq6/GRxAx3V+3/6tOtNzUjN+fPn8dRTTyEhIQH79+/H6tWr8fHHH2uNr3+llJWVFZycnNC7d2+N8UeOHMGiRYvEZVtbW9ja2jaYk0KhAKAcBao5d+bChQtITU3F2rVrxTg/Pz9s27YNvr6+CA8Ph7+/P/7xj380+pnbJEFPI0eOFCIjI1Xa+vXrJ6xYsULnPmbMmCHMnTtXXN61a5dgb2/f7Ou9f/++UFJSIj5u3LghABBKSkp0zrUxZWVlAgABgFBWVtZs/T4sY82LiNqn8vJyITs7WygvLzd0Knq7du2a4OLiIrz++uuCIAjCjz/+KEgkEuHHH3/UuY/g4GBh165dGl+7efOmYG5uLty8eVOvvIqLiwUrKyth7ty5Qk5OjvD5558LXl5ewtKlS1XiPvvsM2HGjBnCxo0bhcWLF+u1jtbU0G+kpKREp7/feh1+qqysREZGBkJCQlTaQ0JCcOrUKZ36yMzMxKlTpxAcHKzSXlZWBk9PT7i5uWHy5MnIzMx86PXGx8fD3t5efLi7u+uUIxEREQD8+eefmDhxIqZOnYp//etfAAAfHx9MmTIFq1atapZ1fPbZZ/Dz84OTk5Ne7+vatSsOHjyIs2fPYvDgwXj++ecRGRmJd955RyVu8uTJ+PXXX5GcnIzNmzc3S87GSq/DT7du3dJ4LNHZ2RlFRUUNvtfNzQ1//PEHqqqqEBMTozLM1q9fP+zevRuDBg1CaWkpNm/ejFGjRuGnn35Cnz59mrzelStXIjo6WlwuLS1lYUNERDpzcHBATk6OWvuRI0f06ufkyZNaXzty5IjKVCH6mDx5MiZPntxgzNmzZ3Hnzh307dsX5uamfX1Qkz5d/bOqBUFo9EzrtLQ0lJWV4cyZM1ixYgV69+6NsLAwAIC/vz/8/f3F2FGjRmH48OHYsmUL3n333SavVyaTQSaT6fy5iIiIWltQUJD497C5FRQUYNGiRThx4gRmzpyJnJwc9O/fv0XWZQz0KmocHR0hlUrVRkeKi4sbPRPcy8sLADBo0CDcvHkTMTExWjeimZkZRowYgdzc3IdeLxERkTF76aWXWqTf8vJyzJo1C1u3boWXlxdeeuklrFu3Dh999FGLrM8Y6HVOjYWFBXx8fJCSkqLSnpKSgsDAQJ37EQQBFRUVDb6elZUlTnLXXOslIiJqL6ysrHD69Gk8+uijAICwsDCTLmiAJhx+io6Oxrx58+Dr64uAgAAkJiYiLy8PkZGRAJTnsRQUFGDv3r0AgG3btsHDwwP9+vUDoJy3ZuPGjVi6dKnYZ2xsLPz9/dGnTx+Ulpbi3XffRVZWFrZt26bzekm7goIC9O3b19BpEBERtSi9i5rQ0FDcvn0ba9euRWFhIQYOHIhjx47B09MTgPIu1nl5eWK8QqHAypUrce3aNZibm6NXr15Yv369ykyJd+7cwXPPPYeioiLY29tj2LBhSE1NxciRI3VeL6nas2eP+Lx///5ITEzEwoULDZgRERFRy5IIgtC0+aLboNLSUtjb26OkpAR2dnbN0qdcLoeNjXKK77KyMqO491N+fj48PT3FiZkAQCqV4vr163BzczNgZkTUXt2/fx/Xrl0TZ4Unqq+h34iuf7+bdJsEMm65ubkqBQ0AVFdX4+rVqwbKiIiIqOWxqDFBffr0Ubt1vVQq1To9NxERkSlgUWOC3NzcsGXLFnFZKpUiISGBh56IiMikmfbUgu1YeHg4oqKiAADZ2dm8+omI2qzCwkIUFhbqHN+tWzdxShBqX1jUtAPdu3c3dApERE2WkJCA2NhYnePXrFmDmJiYlkuIjBaLGiIiMmoREREq90YqLy9HUFAQAOXcZ1ZWVirxHKVpv1jUEBGRUat/OEkul4vPhw4dapCpNMaMGYOhQ4eq3RGbDIsnChMREbWgkydPQiKR4M6dO4ZOxeSxqCEiojaroKDA0CmQEWFRQ0REbUr928Ds3LmzRdcnl8sxf/582NjYoFu3bnjrrbdUXv/www/h6+sLW1tbuLi44Omnn0ZxcTEA4Pr16xg7diwAoHPnzpBIJFiwYAEA4KuvvkJQUBA6deqELl26YPLkyfjll19a9LOYOhY1RETUZuTn56vcEFmhUCAiIgL5+fktts5//vOfOHHiBA4fPozjx4/j5MmTyMjIEF+vrKzEa6+9hp9++gmffvoprl27JhYu7u7u+OSTTwAAly9fRmFhITZv3gxAWSxFR0fj3Llz+Oabb2BmZoYZM2aozQhPuuOJwkRE1GY0dBuYlphgtKysDDt37sTevXsxfvx4AMqRorrrevbZZ8XnPXv2xLvvvouRI0eirKwMNjY2cHBwAAA4OTmhU6dOYuyTTz6psq6dO3fCyckJ2dnZGDhwYLN/lvaAIzVERNRmtPZtYH755RdUVlYiICBAbHNwcMAjjzwiLmdmZmLatGnw9PSEra0txowZAwDIy8trtO+nn34aPXv2hJ2dHby8vHR6H2nHooaIiNqM1r4NjCAIDb4ul8sREhICGxsbfPjhhzh37hwOHz4MQHlYqiFTpkzB7du38f777+OHH37ADz/8oNP7SDsWNURE1KaEh4eLz7Ozs7Fw4cIWW1fv3r3RoUMHnDlzRmz73//+hytXrgAAfv75Z9y6dQvr16/H6NGj0a9fP/Ek4RoWFhYAlIfJaty+fRs5OTl45ZVX8Nhjj6F///743//+12Kfo71gUUNERG1WS98GxsbGBgsXLsQ///lPfPPNN7h48SIWLFggHgLz8PCAhYUFtmzZgl9//RVHjx7Fa6+9ptKHp6cnJBIJPv/8c/zxxx8oKytD586d0aVLFyQmJuLq1av49ttvER0d3aKfpT1gUUNERNSAN998E48++iimTp2KcePGISgoCD4+PgCArl27Yvfu3fj444/h7e2N9evXY+PGjSrv7969O2JjY7FixQo4OztjyZIlMDMzw/79+5GRkYGBAwfihRdewJtvvmmIj2dSJEJjBwxNSGlpKezt7VFSUgI7O7tm6VMul8PGxgaA8ix5Q0zXrYmx5kVE7dP9+/dx7do1eHl5wdLSUq/31r9Lty73fuL9n9qehn4juv795iXdRERk1Bq6S3dNcVMX79LdfrGoISIio1b/Lt2N4ShN+8WihoiIjBoPJ5GueKIwERERmQQWNURE1Gp4XyPSpjl+Gzz8RERELc7CwgJmZmb4/fff0bVrV1hYWEAikRg6LTICgiCgsrISf/zxB8zMzMTJCpuCRQ0REbU4MzMzeHl5obCwEL///ruh0yEj1LFjR3h4eKjd20sfLGqo3ak/50VjeJIiUfOwsLCAh4cHqqqqVG4ZQCSVSmFubv7Qo3csaqjdaWjOC0045wVR85FIJOjQoQM6dOhg6FTIBLGooXan/pwXusxOSkRExo9FDbU79Q8nyeVy8fnQoUN5SwkiojaKl3QTERGRSWBRQ0RERCaBRQ0RERGZBBY1REREZBKaVNRs374dXl5esLS0hI+PD9LS0rTGpqenY9SoUejSpQusrKzQr18/vP322yox77//PkaPHo3OnTujc+fOGDduHM6ePasSExMTA4lEovJwcXFpSvpE1MYVFhbi/PnzOj/0mZeIiNouva9+OnDgAJYvX47t27dj1KhRSEhIwMSJE5GdnQ0PDw+1eGtrayxZsgSDBw+GtbU10tPTERERAWtrazz33HMAgJMnTyIsLAyBgYGwtLTEhg0bEBISgkuXLqF79+5iXwMGDMDXX38tLkul0qZ8ZiJq4zjXEBFpIhEEQdDnDX5+fhg+fDh27NghtvXv3x/Tp09HfHy8Tn3MnDkT1tbW+Pe//63x9erqanTu3Blbt27F/PnzAShHaj799FNkZWXpnGtFRQUqKirE5dLSUri7u6OkpAR2dnY699MQuVwOGxsbAEBZWZnRXA5srHkZI35XbU/9WaF1mWuI8w0RtV2lpaWwt7dv9O+3XiM1lZWVyMjIwIoVK1TaQ0JCcOrUKZ36yMzMxKlTp7Bu3TqtMffu3cODBw/g4OCg0p6bmwtXV1fIZDL4+fkhLi4OPXv21NpPfHy8Xv+bI6K2gXMNEZEmep1Tc+vWLVRXV8PZ2Vml3dnZGUVFRQ2+183NDTKZDL6+voiKisKiRYu0xq5YsQLdu3fHuHHjxDY/Pz/s3bsXycnJeP/991FUVITAwEDcvn1baz8rV65ESUmJ+Lhx44aOn5SIiIjamibNKFz/hlOCIDR6E6q0tDSUlZXhzJkzWLFiBXr37o2wsDC1uA0bNiApKQknT56EpaWl2D5x4kTx+aBBgxAQEIBevXphz549iI6O1rhOmUwGmUymz0cjIiKiNkqvosbR0RFSqVRtVKa4uFht9KY+Ly8vAMqC5ObNm4iJiVErajZu3Ii4uDh8/fXXGDx4cIP9WVtbY9CgQcjNzdXnI7SogoIC9O3b19BpEBERtUt6HX6ysLCAj48PUlJSVNpTUlIQGBiocz+CIKicwAsAb775Jl577TV89dVX8PX1bbSPiooK5OTkGPzkvz179ojP+/fvj507dxowGyIiovZL78NP0dHRmDdvHnx9fREQEIDExETk5eUhMjISgPI8loKCAuzduxcAsG3bNnh4eKBfv34AlFcmbNy4EUuXLhX73LBhA1avXo19+/ahR48e4kiQjY2NeFXKiy++iClTpsDDwwPFxcVYt24dSktLER4e/nDfwEPIz89X+RwKhQIRERGYMGEC3NzcDJYXERFRe6R3URMaGorbt29j7dq1KCwsxMCBA3Hs2DF4enoCUF5qmZeXJ8YrFAqsXLkS165dg7m5OXr16oX169cjIiJCjNm+fTsqKysxa9YslXXVnVsiPz8fYWFhuHXrFrp27Qp/f3+cOXNGXK8h5ObmQqFQqLRVV1fj6tWrLGqIiIhamd7z1LRlul7nrqv8/Hx4enqqFDZSqRTXr183eFHDuVd0x++q7eM2JDJtuv795r2fHoKbmxu2bNkiLkulUiQkJBi8oCEiImqPWNQ8pLrn9GRnZ2PhwoUGzIaIiKj9YlHTjOrep4qIiIhaF4saIiIiMgksaoiIiMgksKghIiIik9Ckez8RUfMqLCxEYWGhzvH171JNREQsaoiMQkJCAmJjY3WOrzsxJRERKbGoITICERERmDp1qrhcXl6OoKAgAMpbi1hZWanEc5SGiEgdixoiI1D/cJJcLhefDx06lDPkEhHpgCcKExERkUngSE0z6SgDUCUHqjS8KJECUsva5Sq5hqAaZoB5nUMNesXeAyCI7+soq9NHlQQw76g5Vj3herHlABRaYgGY1xlF0Ce2+j4gVDdPrLQjIJH8FVsBCJo2hJZYte+qbqwVIPmr9q+uBIQHDfSrR6yZJWAm1R5bN6e6n1vxAFBUNtCvDDAzb0JsFaCoaCDWAjDr0ITYakBxX3uspAMgtdA/VlAA1eW1r9XfhtUNxKr1aw5I/3qzIADV95opVo9/94bYR6gnwX1Ek2INtI/QFmsM+wgD4g0tH1LNjfSEjxoIcn0CGPNF7fIBa+07Q6dgYNzJ2uVPugIVtzTHOvgCj5+rXT7SA5D/pjnW3huYdKl2+YsBQEm25lhrT2Da9drlr0YAf/6oOVbmCDz5R+3y12OA4u80x0o7AqF1dsAnJwG/H9McCwBP1/lppj0F3PiP9tjZZbU7uNMLgGt7tMfOLAYsuyqfn4sCcrdrj516DbDpoXye+U8gZ6P22CcuAp0GKJ//Nwa42MCJvxPOAl1GKJ9nvwlkvaQ1tHzUMVh5TlQuXNkG/LhEe7/BnwPdJymf/7obOPOM9tigg4DHU8rneR8D6bO1x/rvAnouUD4v+AL4brL2WN+tQN8o5fObJ4FvxmqPHboB8P6n8vntc0DySO2xA9cAg2OUz+9cAo4N1B7b/0Vg2JvK52XXgaNe2mP7LAZGbFM+v/8HcMhJe6xXOBCwW/m8Sg4ctNEe6z4LGP1x7fI+ifZY7iOUuI+opcc+Ao+dAJzHKJ8bwz6iBfCGlkRERNSucKTmIdWM1HSUAcU3b2o+odMAQ8tyuRxOzs4AavKy4dCyllh5WUm976rOeg00tKyy/W6VwNrmr9+rMQwtG+HhJ7Xfu00nHn7SGMvDT8pYHn7SP9awh590/fvNc2qayb0KKP+BmWsoaurTJaZJsXV2MuZ/5VTTR93X6sc22q9V4zFNia27E2/WWBkAWaNhYqy5db3vSst3LrUAYKFjvw8ZW3f7SaS17WYddN9x6BVrXrvzatZYKWCm429Yn1iJmep2qv97l1poj22wX0nLxNbkZfBYff7dcx+hf2wr7iO0MYZ9hAEZf4akk/oz0paX1/7PNCsrS+M8Jy091wlnySUiotbEosZENDQjbc0kbnW1xoy0nCWXiIhaE4saE1F/RtrGtMaICGfJJSKi1sSixkQY46EbzpJLREStiZd0ExERkUlgUUNEREQmgUUNERERmQQWNURERGQSWNQQERGRSWBRQ0RERCaBRQ0RERGZBBY1REREZBJY1BAREZFJYFFDREREJoFFDREREZkEFjVERERkEljUENVRUFBg6BSIiKiJmlTUbN++HV5eXrC0tISPjw/S0tK0xqanp2PUqFHo0qULrKys0K9fP7z99ttqcZ988gm8vb0hk8ng7e2Nw4cPP9R6iXS1Z88e8Xn//v2xc+dOA2ZDRERNpXdRc+DAASxfvhyrVq1CZmYmRo8ejYkTJyIvL09jvLW1NZYsWYLU1FTk5OTglVdewSuvvILExEQx5vTp0wgNDcW8efPw008/Yd68eZg9ezZ++OGHJq+XSBf5+flYunSpuKxQKBAREYH8/HwDZkVERE0hEQRB0OcNfn5+GD58OHbs2CG29e/fH9OnT0d8fLxOfcycORPW1tb497//DQAIDQ1FaWkpvvzySzHm8ccfR+fOnZGUlNTk9VZUVKCiokJcLi0thbu7O0pKSmBnZ6f7h26AXC6HjY0NAKCsrAzW1tbN0q8pMsbv6sSJE/jb3/6msX3MmDGtn9BfjPG7Mmb8vohMW2lpKezt7Rv9+63XSE1lZSUyMjIQEhKi0h4SEoJTp07p1EdmZiZOnTqF4OBgse306dNqfU6YMEHss6nrjY+Ph729vfhwd3fXKUdqP/r06QMzM9V/BlKpFL179zZQRkRE1FR6FTW3bt1CdXU1nJ2dVdqdnZ1RVFTU4Hvd3Nwgk8ng6+uLqKgoLFq0SHytqKiowT6but6VK1eipKREfNy4cUOnz0nth5ubG7Zs2SIuS6VSJCQkwM3NzYBZERFRU5g35U0SiURlWRAEtbb60tLSUFZWhjNnzmDFihXo3bs3wsLC9OpT3/XKZDLIZLIG8yIKDw9HVFQUACA7Oxt9+/Y1cEZERNQUehU1jo6OkEqlaqMjxcXFaqMo9Xl5eQEABg0ahJs3byImJkYsalxcXBrs82HWS6SP7t27GzoFekgFBQUsTInaKb0OP1lYWMDHxwcpKSkq7SkpKQgMDNS5H0EQVE7gDQgIUOvz+PHjYp/NtV4iMk28LJ+IgCYcfoqOjsa8efPg6+uLgIAAJCYmIi8vD5GRkQCU57EUFBRg7969AIBt27bBw8MD/fr1A6Cct2bjxo0ql9EuW7YMjz76KN544w1MmzYNR44cwddff4309HSd10tE7ZO2y/InTJjAc6OI2hm9i5rQ0FDcvn0ba9euRWFhIQYOHIhjx47B09MTAFBYWKgyd4xCocDKlStx7do1mJubo1evXli/fj0iIiLEmMDAQOzfvx+vvPIKVq9ejV69euHAgQPw8/PTeb1E1D7l5uZCoVCotFVXV+Pq1assaojaGb3nqWnLdL3OXR+cH0N3xvpdGWNexpiTscrPz4enp6dKYSOVSnH9+nUWNUQmokXmqSEiMja8LJ+IarCoIaI2Lzw8XHyenZ2NhQsXGjAbIjIUFjVEZFJ4WT5R+8WihoiIiEwCixoiIiIyCSxqiIiIyCSwqCEiIiKTwKKGiIiITAKLGiIiIjIJet8mob0rLCxEYWGhuFxeXi4+z8rKgpWVlUp8t27d0K1bt1bLj4iIqL1iUaOnhIQExMbGanwtKChIrW3NmjWIiYlp4azIlBUUFKBv376GToOIyOixqNFTREQEpk6dqnM8R2moKfbs2SM+79+/PxITEzlLLhFRI3hDS2o1xnqTRmPLizdo1J+xbUMial68oSVRG5Wbm6tS0ABAdXU1rl69aqCMiIjaBhY1REamT58+MDNT/acplUrRu3dvA2VERNQ2sKghMjJubm7YsmWLuCyVSpGQkMBDT0REjWBRQ2SEwsPDxefZ2dk8SZiISAe8+qmZyOX6v0cmA8z/2gJVVUBFBWBmBtSd6qYp/VpYAB06KJ9XVwP37wMSCdCxY23MvXuAvqeId+ig7BsAFAqgZoqeuudklpcrX9NE+Vk61nmuZG6u/C4AZU737qn3e/++8rPoQyoFLC3rr1/5PUgkyucVFdrz0kbbNrKyUr4GAJWVwIMH+uVbfxsBVgAkcHHpLrY8eKDsW1+atpGm39/D9FuzjTT9/vRVfxtVVWn//QG6b0NN20jb708flpbK3xtQu420/f700d72EdoYeh9RVaVfv621j6jZRpp+f/pqrn2Ewc/RF9qRkpISAYBQUlLS7H0rf1r6PQ4erH3/wYPKtuBg1X4dHfXvd+vW2vefOKFs8/ZW7dfbW/9+16ypff/Fi8o2R0fVfoOD9e938eLa9xcX17bXNWuW/v3OmqV5GxUX17YtXqx/v9q20cWLtW1r1ujfb91tVFZWJgAXBUAQjh27J7Zv3ap/v9q2kabfn74PTdtI0+9P34embaTp96fvQ9M20vb70+dx4kRtHzXbSNvvT58H9xENb6O62ts+ou420vT70+fRnPuIlqLr328efiIiIiKTwHlqmgmHlpUaPvwkh7OzIwABmZn/RZ8+yllyDT20XFIih7OzEwDg5s3iRuc4aY2hZeW8K10BSFBSUgw7O2VOhh5abqhfwx5+0m0b8vCTkrHuI7Qx9D6Ch58Mf/hJ17/fPKemmTzshjQ3r/3xNGe/UqnmPlTP3dCfmZnmfuvd+kqFcpZc5V84Hx/Ns+RKJJr7rbvjaSpN/cpkNe33xBh9v3NN8RYWtTv3plP+RajZWQHKPxo1f4yaStM20vb704embaTt96cPmaz2D1oNzb8//bahpm2k7fenD23byFD7iPr3q2tMzf3qDLGP0IUh9hH1f3/N0W9z7CM0bSNj3ke0hjaQIpmC/Px8LF26VFxWKBSIiIjAhAkTeKkyUQtq6H51mvB+ddSWsaihVtHQLLksasgUNHVEpKXVv19deXm5ePPd9PR0WNX7bznvV0dtGYsaahU1s+TWv58RZ8klU2GsIyL1iyd5nZNwhg4dyvtkkUlhUUOtomaW3KioKACcJZdMD0dEiAyPRQ21mvDwcLGoyc7ORt++fQ2cEVHz4YgIkeGxqCGD6N69e+NBZHDGep4IEZEmLGqISCtjPU+EiEgTFjVEpBXPEyGitoRFDRFpxfNEiKgt4b2fiIiIyCSwqCEiIiKT0KSiZvv27fDy8oKlpSV8fHyQlpamNfbQoUMYP348unbtCjs7OwQEBCA5OVklZsyYMZBIJGqPSZMmiTExMTFqr7u4uDQlfSIiIjJBehc1Bw4cwPLly7Fq1SpkZmZi9OjRmDhxIvLy8jTGp6amYvz48Th27BgyMjIwduxYTJkyBZmZmWLMoUOHxEtHCwsLcfHiRUilUjz11FMqfQ0YMEAl7sKFC/qmT0RERCZK7xOFN23ahIULF2LRokUAgHfeeQfJycnYsWMH4uPj1eLfeecdleW4uDgcOXIEn332GYYNGwYAcHBwUInZv38/OnbsqFbUmJub6zU6U1FRgYo690ovLS3V+b1ERETUtug1UlNZWYmMjAyEhISotIeEhODUqVM69aFQKHD37l21QqaunTt3Ys6cOWpXVuTm5sLV1RVeXl6YM2cOfv311wbXFR8fD3t7e/Hh7u6uU45ERETU9ug1UnPr1i1UV1fD2dlZpd3Z2RlFRUU69fHWW29BLpdj9uzZGl8/e/YsLl68iJ07d6q0+/n5Ye/evejbty9u3ryJdevWITAwEJcuXUKXLl009rVy5UpER0eLy6WlpS1X2NS51JW0kMvRsc5zo2GMeRljTgDz0ocx5gQYb15kGgw8zUOT5qmRSCQqy4IgqLVpkpSUhJiYGBw5cgROTk4aY3bu3ImBAwdi5MiRKu0TJ04Unw8aNAgBAQHo1asX9uzZo1K41CWTySCTyRrNq1nY2LTOetowawDiLrReYWxIxpiXMeYEMC99GGNOgPHmRSZCEAy6er0OPzk6OkIqlaqNyhQXF6uN3tR34MABLFy4EAcPHsS4ceM0xty7dw/79+8Xz9dpiLW1NQYNGoTc3FzdPwARERGZLL1GaiwsLODj44OUlBTMmDFDbE9JScG0adO0vi8pKQnPPvsskpKSVC7Tru/gwYOoqKjA3LlzG82loqICOTk5GD16tD4foeWUlRk6A6Mnl8vh9FfxW3zzptHMRmuMeRljTgDz0ocx5gQYb15EzUHvw0/R0dGYN28efH19ERAQgMTEROTl5SEyMhKA8jyWgoIC7N27F4CyoJk/fz42b94Mf39/cZTHysoK9vb2Kn3v3LkT06dP13iOzIsvvogpU6bAw8MDxcXFWLduHUpLSxEeHq73h24R3DHo5F7NE2tro/rOjDEvY8wJYF76MMacAOPNi+hh6V3UhIaG4vbt21i7di0KCwsxcOBAHDt2DJ6engCAwsJClTlrEhISUFVVhaioKERFRYnt4eHh2L17t7h85coVpKen4/jx4xrXm5+fj7CwMNy6dQtdu3aFv78/zpw5I66XSFc18xzVKC8vF59nZWVpvEkjb9RIRGT8JIJg4LN6WlFpaSns7e1RUlICOzs7Q6fT7sjlctj8dUJ1WVmZwYa9Y2JiEBsbq3P8mjVrEBMT03IJaWAs31V9zEt3xpgTYLx5ETVE17/fvEs3tTsRERGYOnWqzvEcpSEiahtY1FC7w8NJRESmiXfpJiIiIpPAooaIiIhMAosaIiIiMgksaoiIiMgk8ERhImpzONcQEWnCooaI2pyEhAStcw0FBQWptRliriEian0saoiozeFcQ0SkCYsaImpzeDiJiDThicJERERkEljUEBERkUlgUUNEREQmgefUNBO5XP/3yGSA+V9boKoKqKgAzMyAulejNqVfCwugQwfl8+pq4P59QCIBOnasjbl3D9D3/uwdOij7BgCFAqi5irbuTX7Ly5WvaaL8LB3rPFcyN1d+F4Ayp3v31Pu9f1/5WfQhlQKWlvXXr/weJBLl84oK5XevD23byMpK+RoAVFYCDx7o12/9bQRYAZCofO4HD5R960vTNtL0+2tM/W2oaRtp+v3pS9M20vb704embaTt96cPS0vl7w2o3Ub1P7ex7COU8eYAlD98Y9pHaMN9hJK2baTp96ev5tpHGPym70I7UlJSIgAQSkpKmr1v5U9Lv8fBg7XvP3hQ2RYcrNqvo6P+/W7dWvv+EyeUbd7eqv16e+vf75o1te+/eFHZ5uio2m9wsP79Ll5c+/7i4tr2umbN0r/fWbM0b6Pi4tq2xYv171fbNrp4sbZtzRr9+627jcrKygTgogAIwrFj98T2rVv171fbNtL0+9P3oWkbafr96fvQtI00/f70fWjaRtp+f/o8Tpyo7aNmG02f/kAAIAD4a3vq329L7SOAxWJebWEfER5epnEbafr96fNoy/uIuttI0+9Pn0dz7iNaiq5/v3n4iYiIjNqlS5cMnQK1ERJBEARDJ9FaSktLYW9vj5KSEtjZ2TVr38YytAwY8+EnOZydnQAAN28Ww/qvN3JoWanuNpLL5bCx6QpAgpKSYtjZKb8MQw8t19+GTk61HfPwk/J57eEnORwdbQAAZWVlAPQfl2+Zw09yODt3AlCFsrIyWFpat/o+4vffi1BUVFSnrRzjx48DAKSkfK02I3T37s7o0UN5CT/3EbXL7enwk65/v1nUUKtR/qGu3clbG/zgq/Ey1u/KWPMyRsb6XRljXsaYExkXXf9+8/ATERERmQRe/URkBHiDRiKih8eihsgI8AaNZAgFBQXo27evodMgajYsaoiMAG/QSK1lz5494vP+/fsjMTERCxcuNGBGRM2HRQ2REeDhJNNmLCMi+fn5WLp0qbisUCgQERGBCRMmwM3NzYCZETUPnihMRNQC6o+I7Ny504DZKOXm5kJRb86F6upqXL161UAZETUvFjVERM1M24hIfn6+AbMC+vTpAzMz1d2+VCpF7969DZQRUfNiUUNE1MyMdUTEzc0NW7ZsEZelUikSEhJ46IlMBosaImqSgoICQ6dgtIx5RCQ8PFx8np2dzZOEyaTwRGEi0hmvnNFNzYhIVFQUAOMdEenevbuhU6BG1J/DqjHt/aID3iaBWoymCeVq5lxJT0/nhHJtTH5+Pjw9PVUOq0ilUly/ft3o/lgbg7pT/1++fNkorn4CjPOWBMaYk7GIiYnROoeVJqY6h5Wuf785UkMthhPKmZaGzhNhUdMwjohQU9Wfw0qX/xy2ZyxqqMVwQjnTUnOeSP2RGmM4T4TIVNUfwZbXuS370KFDOapVD4saajE8nGRa2sp5IkTUfvHqJyLSGa+cISJj1qSiZvv27fDy8oKlpSV8fHyQlpamNfbQoUMYP348unbtCjs7OwQEBCA5OVklZvfu3ZBIJGqP+/fvN3m9RNSyeJ4IERkbvYuaAwcOYPny5Vi1ahUyMzMxevRoTJw4EXl5eRrjU1NTMX78eBw7dgwZGRkYO3YspkyZgszMTJU4Ozs78WqZmoelpWWT10tERETti96XdPv5+WH48OHYsWOH2Na/f39Mnz4d8fHxOvUxYMAAhIaG4tVXXwWgHKlZvnw57ty506zrraioQEVFhbhcWloKd3d3XtJN1ES89FZ3xvpdGWNexpiTsWqv35Wul3TrNVJTWVmJjIwMhISEqLSHhITg1KlTOvWhUChw9+5dODg4qLSXlZXB09MTbm5umDx5sspITlPXGx8fD3t7e/Hh7u6uU45ERETU9uhV1Ny6dQvV1dVwdnZWaXd2dkZRUZFOfbz11luQy+WYPXu22NavXz/s3r0bR48eRVJSEiwtLTFq1Cjk5uY+1HpXrlyJkpIS8XHjxg1dPyoRERG1MU26pFsikagsC4Kg1qZJUlISYmJicOTIETg5OYnt/v7+8Pf3F5dHjRqF4cOHY8uWLXj33XebvF6ZTAaZTNZoXkRERNT26VXUODo6QiqVqo2OFBcXq42i1HfgwAEsXLgQH3/8McaNG9dgrJmZGUaMGCGO1DzMeomIiKh90Ovwk4WFBXx8fJCSkqLSnpKSgsDAQK3vS0pKwoIFC7Bv3z5MmjSp0fUIgoCsrCxx4ramrpeIiIjaD70PP0VHR2PevHnw9fVFQEAAEhMTkZeXh8jISADK81gKCgqwd+9eAMqCZv78+di8eTP8/f3F0RYrKyvY29sDAGJjY+Hv748+ffqgtLQU7777LrKysrBt2zad10tERETtm95FTWhoKG7fvo21a9eisLAQAwcOxLFjx+Dp6QlAeWfmunPHJCQkoKqqClFRUeL06oByZtLdu3cDAO7cuYPnnnsORUVFsLe3x7Bhw5CamoqRI0fqvF4iIiJq3/Sep6Yt0/U6dyLSrL3OkdEUxvpdGWNexpiTsWqv31WLzFNDREREZKxY1BAREZFJYFFDREREJoFFDREREZkEFjVERERkEljUEBERkUlgUUNEREQmgUUNERERmQQWNURERGQSWNQQERGRSWBRQ0RERqOgoMDQKVAbxqKGiIgMas+ePeLz/v37Y+fOnQbMhtoyFjVERGQw+fn5WLp0qbisUCgQERGB/Px8A2ZFbRWLGiIiMpjc3FwoFAqVturqaly9etVAGVFbxqKGiIgMpk+fPjAzU/1TJJVK0bt3bwNlRG0ZixoiIjIYNzc3bNmyRVyWSqVISEiAm5ubAbOitopFDRERGVR4eLj4PDs7GwsXLjRgNtSWsaghIiKj0b17d0OnQG2YuaETMBlVcu2vSaSA1FK3WJgB5lZNjL0HQNCWBGDesYmx5QAUWmIBmFs3Lbb6PiBUN0+stCMgkfwVWwEIVc0UawVI/qr9qysB4UHzxJpZAmZS/WMVDwBFZQOxMsDMvAmxVYCiooFYC8CsAwBAagbIOkD529T01dWJhaIaUNzX3q+kAyC10D9WUADV5c0Uaw5IZX/FCkD1veaJra73WYxlH1ElR0dZnfeK29CA+4i6OdXFfYR6rKbtZ2T7CENiUdNcDtpof831CWDMF7XLnzhp3xk6BQPjTtYuH+kBVNzSHOvgCzx+rnb5C29A/pvmWHtvYNKl2uXkEUBJtuZYa09g2vXa5a8fBf78UXOszBF48o/a5ZMTgeLvNMdKOwKhdXbAaU8Cvx/THAsAT9fZoZ6aB9z4j/bY2WW1O7izEcC1PdpjZxYDll2Vz89HA7nbtcdOvQbY9FA+/+8qIGej9tgnLgKdBiifX4oDLsZqj51wFugyQvn88mYg6yXtsY+dAJzHKJ9fTQR+XKI9NvhzoPsk5fPrHwFnntEeG3QQ8HhK+Tz/MJA+W3us/y6g5wJl6oOBL/4J4AtnzbG+W4G+Ucrnf6QB34zV3u/QDYD3P5XP/3ceSB6pPXbgGmBwjPJ5SQ5wbKD22P4vAsPeVD6X5wFHvbTH9lkMjNimfF5xCzjkpD3WKxwI2K18Xn2vwX/3Mtfpqg1Gso+wlv8G+f/VLNfZhgbcR1gDkP8fIK9f03IfoXxeZx9R810BqN1+RraPMCQefiIiIiKTIBEEQdv4oskpLS2Fvb09SkpKYGdn17ydG8vQMg8/te2h5cZiDTy0LJfLYW9nA1kHoPjmTVhbW2uNVfbbfg8/ye/dh429IwCgrKwM1poOr4j9tt4+Qi4vg5Oz8n/4qtvQcPsIuVxem9PtstqcuI9Qi5XL5XB0coYgAP/NykTfvn2Mah/RUnT9+83DT83FXMPOvdVjOzYe06RYq8ZjmhJbdyferLEyAA39BWlqrAUAC8PGmnXQfcehV6x57c6rEdUK4F4FlL/Nxn6fZlLATMffsD6xEjPd/23oFStpvlhpvWVj2UeYC8rtV/Nebe9vzX2EOWpzqov7CLXYPR/twf2/6pD+g3yQmJioerWYEewjDImHn4iIiNoA3lKiccZfdhERUZMVFhaisLBQXC4vrz0Ul5WVBSsr1ZGTbt26oVu3bq2WnzGp/101prW/q4ZuKcHJCpVY1BARNQNjLR4SEhIQG6v5KpugoCC1tjVr1iAmJqaFszJODX1XmrT2d1VzS4m6hQ1vKaGKRQ0RUTMw1uIhIiICU6dO1Tm+vY7SAOrfVXl5ubjt0tPTNRamranmlhJRUcopE3hLCXW8+omIdCaXy2Fjo5xvpaysTPPVT+2UsR+6MGbG+rsyxrzq5nT58mX07dvXwBm1Dl79RETUilikUGvjLSXUsaghIq2M9TwRIiJNWNQQkVbGep4IEZEmLGqISCueZEpEbQmLGiLSioeTiKgtadKMwtu3b4eXlxcsLS3h4+ODtLQ0rbGHDh3C+PHj0bVrV9jZ2SEgIADJyckqMe+//z5Gjx6Nzp07o3Pnzhg3bhzOnj2rEhMTEwOJRKLycHFxaUr6REREZIL0LmoOHDiA5cuXY9WqVcjMzMTo0aMxceJE5OXlaYxPTU3F+PHjcezYMWRkZGDs2LGYMmUKMjMzxZiTJ08iLCwMJ06cwOnTp+Hh4YGQkBAUFBSo9DVgwADxxMXCwkJcuHBB3/SJiIjIROk9T42fnx+GDx+OHTt2iG39+/fH9OnTER8fr1MfAwYMQGhoKF599VWNr1dXV6Nz587YunUr5s+fD0A5UvPpp58iKytL51wrKipQUVF7l7TS0lK4u7tznhoiIiNijPPBAMaZlzHm1Bp0nadGr5GayspKZGRkICQkRKU9JCQEp06d0qkPhUKBu3fvwsHBQWvMvXv38ODBA7WY3NxcuLq6wsvLC3PmzMGvv/7a4Lri4+Nhb28vPtzd3XXKkYiIiNoevYqaW7duobq6Gs7Ozirtzs7OKCoq0qmPt956C3K5HLNnz9Yas2LFCnTv3h3jxo0T2/z8/LB3714kJyfj/fffR1FREQIDA3H79m2t/axcuRIlJSXi48aNGzrlSERERG1Pk65+kkgkKsuCIKi1aZKUlISYmBgcOXIETk5OGmM2bNiApKQknDx5EpaWlmL7xIkTxeeDBg1CQEAAevXqhT179iA6OlpjXzKZDDKZTJePRERERG2cXkWNo6MjpFKp2qhMcXGx2uhNfQcOHMDChQvx8ccfq4zA1LVx40bExcXh66+/xuDBgxvsz9raGoMGDUJubq4+H4GIiIhMlF6HnywsLODj44OUlBSV9pSUFAQGBmp9X1JSEhYsWIB9+/Zh0qRJGmPefPNNvPbaa/jqq6/g6+vbaC4VFRXIycnhHBpEREQEoAmHn6KjozFv3jz4+voiICAAiYmJyMvLQ2RkJADleSwFBQXYu3cvAGVBM3/+fGzevBn+/v7iKI+VlRXs7e0BKA85rV69Gvv27UOPHj3EGBsbG/Es7xdffBFTpkyBh4cHiouLsW7dOpSWliI8PPzhvwUiIiJq8/SepyY0NBTvvPMO1q5di6FDhyI1NRXHjh2Dp6cnAOUN8OrOWZOQkICqqipERUWJs5N269YNy5YtE2O2b9+OyspKzJo1SyVm48aNYkx+fj7CwsLwyCOPYObMmbCwsMCZM2fE9RIREVH7pvc8NW2Zrte5ExFR6zHWuVeMMS9jzKk1tMg8NURERETGikUNERERmQQWNURERGQSWNQQERGRSWBRQ0RERCaBRQ0RERGZBBY1REREZBJY1BAREZFJYFFDREREJoFFDREREZkEvW9oSURE9DAKCwtRWFgoLpeXl4vPs7KyYGVlpRJfcz9AosawqCEiolaVkJCA2NhYja8FBQWpta1ZswYxMTEtnBWZAhY1RETUqiIiIjB16lSd4zlKQ7piUUNERK2Kh5OopfBEYSIiIjIJLGqIiIgaUVBQYOgUSAcsaoiIiDTYs2eP+Lx///7YuXOnAbMhXbCoISIiqic/Px9Lly4VlxUKBSIiIpCfn2/ArKgxLGqIiIjqyc3NhUKhUGmrrq7G1atXDZQR6YJFDRERUT19+vSBmZnqn0ipVIrevXsbKCPSBS/pJiIiqsfNzQ1btmxBVFQUAGVBk5CQADc3t1bNg7Mv60ciCIJg6CRaS2lpKezt7VFSUgI7OztDp0NEREZMLpfDxsYGAHD58mX07du31XOIiYnROvuyJqY6+7Kuf785UkNERNSI7t27G2S9nH1ZPyxqiIiIjFR7P5ykL54oTERERCaBRQ0RERGZBBY1REREZBJY1BAREZFJYFFDREREJoFFDREREZkEFjVERERkEljUEBERkUlgUUNEREQmoUlFzfbt2+Hl5QVLS0v4+PggLS1Na+yhQ4cwfvx4dO3aFXZ2dggICEBycrJa3CeffAJvb2/IZDJ4e3vj8OHDD7VeIiIial/0LmoOHDiA5cuXY9WqVcjMzMTo0aMxceJE5OXlaYxPTU3F+PHjcezYMWRkZGDs2LGYMmUKMjMzxZjTp08jNDQU8+bNw08//YR58+Zh9uzZ+OGHH5q8XiIiImpf9L5Lt5+fH4YPH44dO3aIbf3798f06dMRHx+vUx8DBgxAaGgoXn31VQBAaGgoSktL8eWXX4oxjz/+ODp37oykpKQmr7eiogIVFRXicmlpKdzd3XmXbiIialTdu3SXlZXB2trawBm1X7repVuvkZrKykpkZGQgJCREpT0kJASnTp3SqQ+FQoG7d+/CwcFBbDt9+rRanxMmTBD7bOp64+PjYW9vLz7c3d11ypGIiIjaHr2Kmlu3bqG6uhrOzs4q7c7OzigqKtKpj7feegtyuRyzZ88W24qKihrss6nrXblyJUpKSsTHjRs3dMqRiIiI2h7zprxJIpGoLAuCoNamSVJSEmJiYnDkyBE4OTnp3ae+65XJZJDJZI3mRURERG2fXkWNo6MjpFKp2uhIcXGx2ihKfQcOHMDChQvx8ccfY9y4cSqvubi4NNjnw6yXiIiI2ge9Dj9ZWFjAx8cHKSkpKu0pKSkIDAzU+r6kpCQsWLAA+/btw6RJk9ReDwgIUOvz+PHjYp9NXS8RERG1H3offoqOjsa8efPg6+uLgIAAJCYmIi8vD5GRkQCU57EUFBRg7969AJQFzfz587F582b4+/uLoy1WVlawt7cHACxbtgyPPvoo3njjDUybNg1HjhzB119/jfT0dJ3XS0RERO2c0ATbtm0TPD09BQsLC2H48OHCd999J74WHh4uBAcHi8vBwcECALVHeHi4Sp8ff/yx8MgjjwgdOnQQ+vXrJ3zyySd6rVcXJSUlAgChpKREr/cREVH7U1ZWJv7NKisrM3Q67Zquf7/1nqemLdP1OnciIiLOU2M8WmSeGiIiIiJjxaKGiIiITAKLGiIiIjIJLGqIiIjIJLCoISIiIpPAooaIiIhMAosaIiIiMgksaoiIiMgksKghIiIik8CihoiIiEyC3je0JCIiMkWFhYUoLCwUl8vLy8XnWVlZsLKyUonv1q0bunXr1mr5UeNY1BAREQFISEhAbGysxteCgoLU2tasWYOYmJgWzor0waKGiIgIQEREBKZOnapzPEdpjA+LGiIiIvBwkingicJERERkEljUEBERkUlgUUNEREQmgUUNERERmQQWNURERGQSWNQQERGRSWBRQ0RERCaBRQ0RERGZBBY1REREZBJY1BAREZFJYFFDREREJoFFDREREZkEFjVERERkEtrVXboFQQAAlJaWGjgTIiIi0lXN3+2av+PatKui5u7duwAAd3d3A2dCRERE+rp79y7s7e21vi4RGit7TIhCocDvv/8OW1tbSCQSQ6fT4kpLS+Hu7o4bN27Azs7O0OkYNX5XuuN3pTt+V7rjd6W79vhdCYKAu3fvwtXVFWZm2s+caVcjNWZmZnBzczN0Gq3Ozs6u3fzwHxa/K93xu9Idvyvd8bvSXXv7rhoaoanBE4WJiIjIJLCoISIiIpPAosaEyWQyrFmzBjKZzNCpGD1+V7rjd6U7fle643elO35X2rWrE4WJiIjIdHGkhoiIiEwCixoiIiIyCSxqiIiIyCSwqCEiIiKTwKLGxMTHx2PEiBGwtbWFk5MTpk+fjsuXLxs6rTYhPj4eEokEy5cvN3QqRqugoABz585Fly5d0LFjRwwdOhQZGRmGTsvoVFVV4ZVXXoGXlxesrKzQs2dPrF27FgqFwtCpGVxqaiqmTJkCV1dXSCQSfPrppyqvC4KAmJgYuLq6wsrKCmPGjMGlS5cMk6yBNfRdPXjwAC+//DIGDRoEa2truLq6Yv78+fj9998Nl7ARYFFjYr777jtERUXhzJkzSElJQVVVFUJCQiCXyw2dmlE7d+4cEhMTMXjwYEOnYrT+97//YdSoUejQoQO+/PJLZGdn46233kKnTp0MnZrReeONN/Dee+9h69atyMnJwYYNG/Dmm29iy5Ythk7N4ORyOYYMGYKtW7dqfH3Dhg3YtGkTtm7dinPnzsHFxQXjx48X793XnjT0Xd27dw/nz5/H6tWrcf78eRw6dAhXrlzB1KlTDZCpERHIpBUXFwsAhO+++87QqRitu3fvCn369BFSUlKE4OBgYdmyZYZOySi9/PLLQlBQkKHTaBMmTZokPPvssyptM2fOFObOnWugjIwTAOHw4cPiskKhEFxcXIT169eLbffv3xfs7e2F9957zwAZGo/635UmZ8+eFQAIv/32W+skZYQ4UmPiSkpKAAAODg4GzsR4RUVFYdKkSRg3bpyhUzFqR48eha+vL5566ik4OTlh2LBheP/99w2dllEKCgrCN998gytXrgAAfvrpJ6Snp+OJJ54wcGbG7dq1aygqKkJISIjYJpPJEBwcjFOnThkws7ahpKQEEomkXY+etqsbWrY3giAgOjoaQUFBGDhwoKHTMUr79+/H+fPnce7cOUOnYvR+/fVX7NixA9HR0fjXv/6Fs2fP4vnnn4dMJsP8+fMNnZ5Refnll1FSUoJ+/fpBKpWiuroar7/+OsLCwgydmlErKioCADg7O6u0Ozs747fffjNESm3G/fv3sWLFCjz99NPt6iaX9bGoMWFLlizBf//7X6Snpxs6FaN048YNLFu2DMePH4elpaWh0zF6CoUCvr6+iIuLAwAMGzYMly5dwo4dO1jU1HPgwAF8+OGH2LdvHwYMGICsrCwsX74crq6uCA8PN3R6Rk8ikagsC4Kg1ka1Hjx4gDlz5kChUGD79u2GTsegWNSYqKVLl+Lo0aNITU2Fm5ubodMxShkZGSguLoaPj4/YVl1djdTUVGzduhUVFRWQSqUGzNC4dOvWDd7e3ipt/fv3xyeffGKgjIzXP//5T6xYsQJz5swBAAwaNAi//fYb4uPjWdQ0wMXFBYByxKZbt25ie3FxsdroDSk9ePAAs2fPxrVr1/Dtt9+261EagFc/mRxBELBkyRIcOnQI3377Lby8vAydktF67LHHcOHCBWRlZYkPX19f/P3vf0dWVhYLmnpGjRqlNj3AlStX4OnpaaCMjNe9e/dgZqa6e5VKpbykuxFeXl5wcXFBSkqK2FZZWYnvvvsOgYGBBszMONUUNLm5ufj666/RpUsXQ6dkcBypMTFRUVHYt28fjhw5AltbW/EYtb29PaysrAycnXGxtbVVO9fI2toaXbp04TlIGrzwwgsIDAxEXFwcZs+ejbNnzyIxMRGJiYmGTs3oTJkyBa+//jo8PDwwYMAAZGZmYtOmTXj22WcNnZrBlZWV4erVq+LytWvXkJWVBQcHB3h4eGD58uWIi4tDnz590KdPH8TFxaFjx454+umnDZi1YTT0Xbm6umLWrFk4f/48Pv/8c1RXV4v7ewcHB1hYWBgqbcMy8NVX1MwAaHzs2rXL0Km1Cbyku2GfffaZMHDgQEEmkwn9+vUTEhMTDZ2SUSotLRWWLVsmeHh4CJaWlkLPnj2FVatWCRUVFYZOzeBOnDihcR8VHh4uCILysu41a9YILi4ugkwmEx599FHhwoULhk3aQBr6rq5du6Z1f3/ixAlDp24wEkEQhNYsooiIiIhaAs+pISIiIpPAooaIiIhMAosaIiIiMgksaoiIiMgksKghIiIik8CihoiIiEwCixoiIiIyCSxqiIiIyCSwqCEiagW7d+9Gp06dWnw9lZWV6N27N77//nsAwPXr1yGRSJCVlQUAuHDhAtzc3CCXy1s8F6LWxqKGyAgtWLAAEokEkZGRaq8tXrwYEokECxYsaP3ETIxEIsGnn37abHHGIDExEZ6enhg1ahQAwN3dHYWFheL9zAYNGoSRI0fi7bffNmSaRC2CRQ2RkXJ3d8f+/ftRXl4utt2/fx9JSUnw8PAwYGa6qaysNHQK7dKWLVuwaNEicVkqlcLFxQXm5rX3L37mmWewY8cOVFdXGyJFohbDoobISA0fPhweHh44dOiQ2Hbo0CG4u7tj2LBhKrGCIGDDhg3o2bMnrKysMGTIEPznP/8RX6+ursbChQvh5eUFKysrPPLII9i8ebNKHydPnsTIkSNhbW2NTp06YdSoUfjtt98AKEeOpk+frhK/fPlyjBkzRlweM2YMlixZgujoaDg6OmL8+PEAgOzsbDzxxBOwsbGBs7Mz5s2bh1u3bqm8b+nSpVi+fDk6d+4MZ2dnJCYmQi6X45lnnoGtrS169eqFL7/8UmX9uvT7/PPP46WXXoKDgwNcXFwQExMjvt6jRw8AwIwZMyCRSMTlxtQczjl06BDGjh2Ljh07YsiQITh9+rRK3O7du+Hh4YGOHTtixowZuH37tlpfn332GXx8fGBpaYmePXsiNjYWVVVVAIC1a9fC1dVV5X1Tp07Fo48+CoVCoTG38+fP4+rVq5g0aZJavjWHnwBgwoQJuH37Nr777judPjNRW8GihsiIPfPMM9i1a5e4/H//93949tln1eJeeeUV7Nq1Czt27MClS5fwwgsvYO7cueIfLYVCATc3Nxw8eBDZ2dl49dVX8a9//QsHDx4EAFRVVWH69OkIDg7Gf//7X5w+fRrPPfccJBKJXvnu2bMH5ubm+P7775GQkIDCwkIEBwdj6NCh+PHHH/HVV1/h5s2bmD17ttr7HB0dcfbsWSxduhT/+Mc/8NRTTyEwMBDnz5/HhAkTMG/ePNy7dw8A9OrX2toaP/zwAzZs2IC1a9ciJSUFAHDu3DkAwK5du1BYWCgu62rVqlV48cUXkZWVhb59+yIsLEwsSH744Qc8++yzWLx4MbKysjB27FisW7dO5f3JycmYO3cunn/+eWRnZyMhIQG7d+/G66+/Lvbfo0cPcdTlvffeQ2pqKv7973/DzEzzrjs1NRV9+/aFnZ1dg7lbWFhgyJAhSEtL0+szExk9A98lnIg0CA8PF6ZNmyb88ccfgkwmE65duyZcv35dsLS0FP744w9h2rRpQnh4uCAIglBWViZYWloKp06dUulj4cKFQlhYmNZ1LF68WHjyyScFQRCE27dvCwCEkydPNphPXcuWLROCg4PF5eDgYGHo0KEqMatXrxZCQkJU2m7cuCEAEC5fviy+LygoSHy9qqpKsLa2FubNmye2FRYWCgCE06dPN7lfQRCEESNGCC+//LK4DEA4fPiwxs9cV924a9euCQCEDz74QHz90qVLAgAhJydHEARBCAsLEx5//HGVPkJDQwV7e3txefTo0UJcXJxKzL///W+hW7du4vIvv/wi2NraCi+//LLQsWNH4cMPP2wwz2XLlgl/+9vfVNpq8s3MzFRpnzFjhrBgwYIG+yNqa8y1VjtEZHCOjo6YNGkS9uzZA0EQMGnSJDg6OqrEZGdn4/79++LhnhqVlZUqh6nee+89fPDBB/jtt99QXl6OyspKDB06FADg4OCABQsWYMKECRg/fjzGjRuH2bNno1u3bnrl6+vrq7KckZGBEydOwMbGRi32l19+Qd++fQEAgwcPFtulUim6dOmCQYMGiW3Ozs4AgOLi4ib3CwDdunUT+3hYdfuu+Z6Ki4vRr18/5OTkYMaMGSrxAQEB+Oqrr8TljIwMnDt3ThyZAZSHCe/fv4979+6hY8eO6NmzJzZu3IiIiAiEhobi73//e4M5lZeXw9LSUqf8raysxJEvIlPBoobIyD377LNYsmQJAGDbtm1qr9ecX/HFF1+ge/fuKq/JZDIAwMGDB/HCCy/grbfeQkBAAGxtbfHmm2/ihx9+EGN37dqF559/Hl999RUOHDiAV155BSkpKfD394eZmRkEQVDp+8GDB2q5WFtbq+U2ZcoUvPHGG2qxdQumDh06qLwmkUhU2moOg9V81ofpV9v5KPpqKL/635UmCoUCsbGxmDlzptprdQuT1NRUSKVSXL9+HVVVVSon/Nbn6OiICxcu6JT/n3/+iV69eukUS9RWsKghMnKPP/64eCXRhAkT1F739vaGTCZDXl4egoODNfaRlpaGwMBALF68WGz75Zdf1OKGDRuGYcOGYeXKlQgICMC+ffvg7++Prl274uLFiyqxWVlZakVDfcOHD8cnn3yCHj16NPjHWF/N1W+HDh1a5Aogb29vnDlzRqWt/vLw4cNx+fJl9O7dW2s/Bw4cwKFDh3Dy5EmEhobitddeQ2xsrNb4YcOGYceOHRAEodHzoS5evIhZs2bp8GmI2g6eKExk5KRSKXJycpCTkwOpVKr2uq2tLV588UW88MIL2LNnD3755RdkZmZi27Zt2LNnDwCgd+/e+PHHH5GcnIwrV65g9erVKifGXrt2DStXrsTp06fx22+/4fjx47hy5Qr69+8PAPjb3/6GH3/8EXv37kVubi7WrFmjVuRoEhUVhT///BNhYWE4e/Ysfv31Vxw/fhzPPvvsQxUTzdVvjx498M0336CoqAj/+9//mpxPfTUjXhs2bMCVK1ewdetWlUNPAPDqq69i7969iImJwaVLl5CTkyOOkAFAfn4+/vGPf+CNN95AUFAQdu/ejfj4eLXiqK6xY8dCLpfj0qVLDeZ3/fp1FBQUYNy4cQ//YYmMCIsaojbAzs6uwStaXnvtNbz66quIj49H//79MWHCBHz22Wfw8vICAERGRmLmzJkIDQ2Fn58fbt++rTJq07FjR/z888948skn0bdvXzz33HNYsmQJIiIiAChHiFavXo2XXnoJI0aMwN27dzF//vxG83Z1dcX333+P6upqTJgwAQMHDsSyZctgb2+v9QoeXTRXv2+99RZSUlI0Xib/MPz9/fHBBx9gy5YtGDp0KI4fPy4WKzUmTJiAzz//HCkpKRgxYgT8/f2xadMmeHp6QhAELFiwACNHjhQPPY4fPx5LlizB3LlzUVZWpnG9Xbp0wcyZM/HRRx81mF9SUhJCQkLg6enZPB+YyEhIBF0O/hIRUZtw4cIFjBs3DlevXoWtra3a6xUVFejTpw+SkpLEWYeJTAWLGiIiE7Nnzx4MHz5c5QqyGleuXMGJEyfEUTgiU8KihoiIiEwCz6khIiIik8CihoiIiEwCixoiIiIyCSxqiIiIyCSwqCEiIiKTwKKGiIiITAKLGiIiIjIJLGqIiIjIJLCoISIiIpPw/x7aBZeKXVtMAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "mu_tot = np.concatenate((mu_prec, mu_osc))\n",
    "SDOM_analysis(len(mu_tot), mu_tot, mu_tot/10)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
