import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def getPeriod2(df, begin, end, resetIndex = False):
    """
    Returns the df in the chosen interval
    
    Object begin: Start date forrmated as 'yyyy.mm.dd'.
    Object   end: End date 'yyyy.mm.dd'.

    returns a dataframe with the historic of the selected period
    """
    
    indexBegin = df[df['date']==begin].index[0]
    indexEnd = df[df['date']==end].tail(1).index[0]
    
    if (resetIndex):
        return df[(df.index >= indexBegin) & (df.index <= indexEnd)].reset_index(drop=True)
    else: 
        return df[(df.index >= indexBegin) & (df.index <= indexEnd)]

def getDay(df, day):
    return df[df['date']==day]

def buy(value, availableMoney, opr='normal'):
    lotValue = value*100
    quantity = math.floor(availableMoney/lotValue)
    
    amount = quantity*lotValue
    remainingMoney = availableMoney - amount
    
    return remainingMoney, quantity

def sell(value, amount, opr='normal'):
    if(opr == 'normal'):
        return value*amount*100
    else:
        return value*amount*100
def AnnualReturn(initial, final, days):
    return ((final/initial)**(365.25/days))-1

#Final = df.tail(1).Amount
#size = len(testVale)
#size = 365
#print(AnnualReturn(100000,Final,size))

def calculateDrawdown(df):
    length = len(df.index)
    lst = []

    for i in range (0, length-1):
        trade = df.iloc[i]
        if (i==0):
            max = trade.Amount
            max_date = trade.date
            max_day = i

            min = trade.Amount
            min_date = trade.date
            min_day = i

        elif(trade.Amount > df.iloc[i-1].Amount and trade.Amount > df.iloc[i+1].Amount):
            if(trade.Amount > max):
                max = trade.Amount
                min = trade.Amount
                max_date = trade.date
                max_day = trade.day
                min_date = trade.date
                min_day = trade.day

                #print(str(trade.day)+' - max ' + str(max))

        elif(trade.Amount < df.iloc[i-1].Amount and trade.Amount < df.iloc[i+1].Amount):

            if(trade.Amount < min):
                min = trade.Amount
                min_date = trade.date
                min_day = trade.day
                #print(str(trade.day)+' - min ' + str(min))
                drawdown = ((max-min)/max)*100

                lst.append([max_day, max_date, max, min_day, min_date, min, drawdown])
                #print(lst)
                #print(drawdown)

    DDDf = pd.DataFrame(lst)
    DDDf.columns = ['maxDay', 'maxDate', 'maxValue','minDay', 'minDate', 'minValue', 'drawdown']
    
    return DDDf, DDDf['drawdown'].max()


def printStrategyResult(petrBR, petrAR, petrDDmax, petrMeanTrades):
    print('Brute Return: '+str(petrBR) + 
        '\nAnnual Return:' +str(petrAR)+
        '\nMax DD: '+ str(petrDDmax)+
        '\nMean Trades: '+ str(petrMeanTrades)+
        '\n#####################################'
        )


def printAmountChart(petrOrders, petrDaily):
    plt.figure(figsize=(10,5),dpi = 160)
    correctTrades = petrOrders[petrOrders['profit'] > 0]
    correctTradesMean = correctTrades['profit'].mean()
    wrongTrades = petrOrders[petrOrders['profit'] < 0]
    wrongTradesMean = wrongTrades['profit'].mean()
    
    print(f'Media de lucro em casos de trades corretos: {correctTradesMean}')
    print(f'Media de prejuízos em casos de trades errados: {wrongTradesMean}')
    print('Accuracy( percentagem das operações que tiveram lucros): '+str(round(100*len(correctTrades.index)/len(petrOrders.index), 2)))
    plt.plot(petrDaily.index, petrDaily['Amount'], label = "Random Forest", color='tab:orange')
    plt.title('PETR4')
    plt.legend()
    plt.xlabel("Dias executados (01/06/2020 - 01/06/2021)")
    plt.ylabel("Valor acumulado")
    # plt.savefig('../../Images/General_petr4.png')
    plt.show()
    plt.clf
                
def makeImgDaily(currentDay, day, dayDf,buyPoints, sellPoints, stopPoints, profitPoints, amountPerDay, money, pathImgs):
    money = round(money,2)
    amountPerDay = round(amountPerDay,2)
    trade = round(100*amountPerDay/money, 2)
    plt.figure(figsize=(10,5),dpi = 160)    
    plt.plot(dayDf.time, dayDf['close'], label = "Valor atual", color='tab:orange',)
    plt.plot(dayDf.time, dayDf['high-pred'], label = "high-pred", color='tab:green')
    plt.plot(dayDf.time, dayDf['low-pred'], label = "low-pred", color='tab:red')
    plt.plot(dayDf.time, dayDf['high-pred-regressor'], label = "high-pred-regressor", color='tab:green', linestyle='dotted')
    plt.plot(dayDf.time, dayDf['low-pred-regressor'], label = "low-pred-regressor", color='tab:red', linestyle='dotted')
    if buyPoints.size > 0:
        plt.plot(buyPoints[: ,0], buyPoints[: ,1].astype(float),label = "buy", color='black', marker='^', linestyle='')
    if sellPoints.size > 0:
        plt.plot(sellPoints[: ,0], sellPoints[: ,1].astype(float), label = "sell", color='black', marker='v', linestyle='')
    if stopPoints.size > 0:
        plt.plot(stopPoints[: ,0], stopPoints[: ,1].astype(float), label = "stop", color='red', marker='.', linestyle='')
    if profitPoints.size > 0:
        plt.plot(profitPoints[: ,0], profitPoints[: ,1].astype(float), label = "profit", color='green', marker='+', linestyle='')
    title = f'Data: {currentDay} Dia: {day} Trade: R${amountPerDay} ({trade}%) Carteira:  {money}'
    plt.title(title)
    plt.legend()
    plt.xlabel("Hora")
    plt.ylabel("Valor do ativo (close)")
    plt.grid()
    
    xValues = []
    for item in dayDf.time.values:
        if item[-2:] == '00':
            xValues.append(item)
    plt.xticks(xValues)
    # plt.yticks(size = 3)
    # plt.margins(y=90)
    plt.savefig(f'{pathImgs}/{title}.png')
    # plt.show()
    plt.close()

def appendPoints(numpyArray, point):
    if(numpyArray.size==0):
        numpyArray = np.append(numpyArray,point)
        numpyArray = numpyArray.reshape((1,len(point)))
    else:
        numpyArray = np.append(numpyArray,[point], axis=0)
    return numpyArray
    
def runStrategy(df, dailyDf, Money, stopLoss,errorMax,errorMin, stdMax, stdMin, alfa, beta, method):
    
    datetime.timestamp(datetime.now())
    timestamp = datetime.timestamp(datetime.now())
    pathDebug = f'./debug_{timestamp}'
    os.mkdir(pathDebug)
    
    size = len(df.index)
    
    initialMoney = Money
    Money = Money
    stopLoss = stopLoss

    orders = []
    daysObserved = []
    
    for day in range(0,size):
        currentDay =  df.iloc[day].date
        lowPredRegressor = df.iloc[day].pred_low
        highPredRegressor = df.iloc[day].pred_high
        highPred, highPred = None, None
        if method=='adrion':
            highPred = highPredRegressor*(1-(errorMax+stdMax))
            lowPred = lowPredRegressor*(1+(errorMin+stdMin))
        else:
            D = highPredRegressor - lowPredRegressor
            highPred = highPredRegressor-(alfa*D)
            lowPred = lowPredRegressor+(beta*D)

        # obtém todos os tempos( 15 em 15 min) do dia currentDay
        dayDf = getPeriod2(dailyDf, currentDay, currentDay)
        dayDf['high-pred-regressor'] = highPredRegressor
        dayDf['low-pred-regressor'] = lowPredRegressor
        dayDf['high-pred'] = highPred
        dayDf['low-pred'] = lowPred
        #
        daySize = len(dayDf.index) 

        op = 'none'
        quantity = 0
        trade = [] # essa variável não está sendo usada, verificar se podemos apagá-la
        tradesPerDay = 0
        amountPerDay = 0
        isStop = False
        countStop = 0
        
        buyPoints = np.array([])
        sellPoints = np.array([])
        stopPoints = np.array([])
        profitPoints = np.array([])

        for i in range(0,daySize):
            currentTime = dayDf.iloc[i]
            if(countStop >= 3):
                countStop = countStop
                break
            if((op == 'buy') and (currentTime.close < opPrice*(1-stopLoss) or currentTime.close >= highPred or i == daySize-1)):
                isStop = currentTime.close < opPrice*(1-stopLoss)
                if(isStop): 
                    countStop = countStop+1
                    stopPoints = appendPoints(stopPoints,[currentTime.time,currentTime.close])
                else:
                    profitPoints = appendPoints(profitPoints,[currentTime.time,currentTime.close])
                aux = sell(currentTime.close, quantity)
                Money = Money + aux 
                trade = trade + [currentTime.time, 'Sell', currentTime.close, currentTime.close-opPrice,(currentTime.close-opPrice)*100*quantity, isStop, Money, highPred, lowPred]
                orders.append(trade)
                
                amountPerDay = amountPerDay + (currentTime.close-opPrice)*100*quantity
                
                quantity = 0
                op = 'none'
                
            elif((op == 'sell') and (currentTime.close > opPrice*(1+stopLoss) or currentTime.close <= lowPred or i == daySize-1)):
                isStop = currentTime.close > opPrice*(1+stopLoss)
                if(isStop): 
                    countStop = countStop+1
                    stopPoints = appendPoints(stopPoints,[currentTime.time,currentTime.close])
                else:
                    profitPoints = appendPoints(profitPoints,[currentTime.time,currentTime.close])
                aux = sell(currentTime.close, quantity)
                Money = Money + (opPrice*100*quantity) + ((opPrice*100*quantity) - aux) 
                trade = trade + [currentTime.time, 'Buy', currentTime.close, opPrice-currentTime.close,(opPrice-currentTime.close)*100*quantity, isStop, Money, highPred, lowPred]
                orders.append(trade)

                amountPerDay = amountPerDay + (opPrice-currentTime.close)*100*quantity
                quantity = 0
                op = 'none'

            elif(op == 'none' and i < daySize-1 ):
                if(currentTime.close <= lowPred):
                    buyPoints = appendPoints(buyPoints,[currentTime.time,currentTime.close])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(currentTime.close, Money)
                    opPrice = currentTime.close
                    op = 'buy'
                    trade = [currentTime.date, quantity, currentTime.time, 'Buy', currentTime.close]
                elif(currentTime.close >=  highPred):
                    sellPoints = appendPoints(sellPoints,[currentTime.time,currentTime.close])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(currentTime.close, Money)
                    opPrice = currentTime.close
                    op = 'sell'
                    trade = [currentTime.date, quantity, currentTime.time, 'Sell', currentTime.close]
        if day%1 == 0:
            makeImgDaily(currentTime.date, day, dayDf, buyPoints, sellPoints, stopPoints, profitPoints, amountPerDay, Money, pathDebug)            
        daysObserved.append([day, currentTime.date, tradesPerDay, amountPerDay, round(Money,2)])
    
    daysDf = pd.DataFrame(daysObserved)
    daysDf.columns = ['day', 'date', 'trades','profit', 'Amount']
    
    ordersDf = pd.DataFrame(orders)
    ordersDf.columns = ['date','batches','time_op1','op1','value_op1','time_op2','op2','value_op2', 'profit', 'totalProfit', 'stop','Amount','max','min']
    
    DDDf, DDmax = calculateDrawdown(daysDf)
    finalMoney = daysDf['Amount'].tail(1).reset_index(drop=True)[0]
    annualReturn = round(AnnualReturn(initialMoney,finalMoney,365)*100,2)
    tradesMean = daysDf['trades'].mean()
    
    
    return ordersDf, daysDf, DDDf, round(DDmax,2), annualReturn, finalMoney, round(tradesMean,2)