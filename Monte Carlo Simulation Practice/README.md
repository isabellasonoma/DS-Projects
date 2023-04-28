## Monte Carlo with Equities
This project allowed me to practice using a Monte Carlo Simulation. I used equity data from the Yahoo website.

First, I found the daily closing price for a stock and plotted the price for a certain time period.
Then, I found the log returns of each day to see how the price changed over a period of time.<br/>
This data allowed me to do a Monte Carlo Simulation. There were two elements of this simulation: drift and randomness.<br/>
I looked at different equities with this simulation. <br/>
### Oracle<br/>
![orcl_price](https://user-images.githubusercontent.com/109932048/235114615-3db5b57b-2093-4312-aa9d-bce2bf53fa77.png)<br/>
![orcl_logreturns](https://user-images.githubusercontent.com/109932048/235114584-3c0ccdd7-dc0c-419f-8f19-eb4aa38df670.png)<br/>
![orcl_monty](https://user-images.githubusercontent.com/109932048/235114627-2a71a890-45a7-4f33-9824-66e02238dbfb.png)<br/>
Expected price:  80.64<br/>
Quantile (5%)  56.54837536918634<br/>
Quantile (95%)  113.99522812518204<br/>
### Mercadolibre<br/>
![meli_price](https://user-images.githubusercontent.com/109932048/235117554-dcdf2b6d-5231-40a3-a0e6-1146e5da25ae.png)
![meli_logreturns](https://user-images.githubusercontent.com/109932048/235117552-56c05d8e-2dfe-4def-ac02-beb64fb7f648.png)
![meli_monty](https://user-images.githubusercontent.com/109932048/235117546-dad50838-b8c3-4130-91e7-f4bcc6128d99.png)<br/>
Expected price:  1228.78<br/>
Quantile (5%)  611.4406075624938<br/>
Quantile (95%)  2263.46230993108<br/>
