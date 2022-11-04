import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

        elif(trade.Amount >= df.iloc[i-1].Amount and trade.Amount > df.iloc[i+1].Amount):
            if(trade.Amount > max):
                max = trade.Amount
                min = trade.Amount
                max_date = trade.date
                max_day = trade.day
                min_date = trade.date
                min_day = trade.day

                #print(str(trade.day)+' - max ' + str(max))

        elif(trade.Amount <= df.iloc[i-1].Amount and trade.Amount < df.iloc[i+1].Amount):

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
    if(len(lst)==0):
        return DDDf, 0
    DDDf.columns = ['maxDay', 'maxDate', 'maxValue','minDay', 'minDate', 'minValue', 'drawdown']
    
    return DDDf, DDDf['drawdown'].max()

def printMeanStrategyResult(finalMoneyList, annualReturnList, DDmaxList, tradesMeanList):
    finalMoneyNp = np.array(finalMoneyList)
    annualReturnNp = np.array(annualReturnList)
    DDmaxNp = np.array(DDmaxList)
    tradesMeanNp = np.array(tradesMeanList)
    print("Mostrando média das estatística")
    printStrategyResult(finalMoneyNp.mean(), annualReturnNp.mean(), DDmaxNp.mean(), tradesMeanNp.mean())
    print('finalMoneyNp', finalMoneyNp)
    print('annualReturnNp', annualReturnNp)
    print('DDmaxNp', DDmaxNp)
    print('tradesMeanNp', tradesMeanNp)


def printStrategyResult(petrBR, petrAR, petrDDmax, petrMeanTrades):
    print('Brute Return: '+str(petrBR) + 
        '\nAnnual Return:' +str(petrAR)+
        '\nMax DD: '+ str(petrDDmax)+
        '\nMean Trades: '+ str(petrMeanTrades)+
        '\n-----------------------------'
        )


def printAmountChart(petrOrders, petrDaily, title, DDDf = None):
    
    # title = 'DrawnDown'
    
    plt.figure(figsize=(10,5),dpi = 160)
    
    correctTrades = petrOrders[petrOrders['profit'] > 0]
    correctTradesMean = correctTrades['totalProfit'].mean()
    wrongTrades = petrOrders[petrOrders['profit'] < 0]
    wrongTradesMean = wrongTrades['totalProfit'].mean()

    
    print(f'Media de lucro em casos de trades corretos: {correctTradesMean}')
    print(f'Media de prejuízos em casos de trades errados: {wrongTradesMean}')
    print('Accuracy( percentagem das operações que tiveram lucros): '+str(round(100*len(correctTrades.index)/len(petrOrders.index), 2)))
    
    
    if type(DDDf) == type(pd.DataFrame()) and not(DDDf.empty):
        plt.plot(
            [
                DDDf['maxDay'],
                DDDf['minDay']
            ],
            [
                DDDf['maxValue'],
                DDDf['minValue']    
            ],
            marker = 'o',
            color='#C2BEBD',
            # label='Todos os drawdown'
        )
        indexMaxDrawdown = DDDf['drawdown'].argmax()
        plt.plot(
            [
                DDDf['maxDay'][indexMaxDrawdown],
                DDDf['minDay'][indexMaxDrawdown]
            ],
            [
                DDDf['maxValue'][indexMaxDrawdown],
                DDDf['minValue'][indexMaxDrawdown]    
            ],
            marker = 'o',
            color='#000',
            label='Maior drawdown'
        )
        

    
    
    plt.plot(petrDaily.index, petrDaily['Amount'], label = "Variação do investimento", color='tab:orange')
    plt.title(title)
    plt.legend()
    plt.xlabel("Dias executados (01/06/2020 - 01/06/2021)")
    plt.ylabel("Valor acumulado")
    plt.savefig(f'./Rentabilidade Anual/{title}.png')
    # plt.show()
    # plt.close()
                
def makeImgDaily(currentDay, day, dayDf,buyPoints, sellPoints, stopPoints, profitPoints, amountPerDay, money, pathImgs, buyPointsDeniedByEngulfing, sellPointsDeniedByEngulfing, maxError=None, minError=None):
    money = round(money,2)
    amountPerDay = round(amountPerDay,2)
    trade = round(100*amountPerDay/money, 2)
    times = pd.to_datetime(dayDf.time)
    plt.figure(figsize=(10,5),dpi = 400)
    ax = plt.axes()
    plt.plot(times, dayDf['close'], label = "Valor atual", color='tab:orange', lw=1)
    
    plt.hlines(y=dayDf['open'], xmin=times-np.timedelta64(2,'m'), xmax=times, colors='teal', ls='-', lw=2)
    plt.hlines(y=dayDf['close'], xmin=times, xmax=times+np.timedelta64(2,'m'), colors='teal', ls='-', lw=2)
    plt.vlines(x=times, ymin=dayDf['low'], ymax=dayDf['high'], colors='teal', ls='-', lw=1)
    
    
    plt.plot(times, dayDf['high-pred'], label = "high-pred", color='tab:green')
    if maxError:
        y = dayDf['high-pred']
        error = maxError*y
        error2 = 0.75*maxError*y
        plt.fill_between(times, y-error, y+error,hatch='+', color ='green')
        plt.fill_between(times, y-error2, y+error2,hatch='+', color ='springgreen')
    
    plt.plot(times, dayDf['low-pred'], label = "low-pred", color='tab:red')
    if minError:
        y = dayDf['low-pred'].values
        error = minError*y
        error2 = 0.75*minError*y
        plt.fill_between(times, y-error, y+error,hatch='+', color ='red')
        plt.fill_between(times, y-error2, y+error2,hatch='+', color ='firebrick')    
    # plt.plot(times, dayDf['high-pred-regressor'], label = "high-pred-regressor", color='tab:green', linestyle='dotted')
    # plt.plot(times, dayDf['low-pred-regressor'], label = "low-pred-regressor", color='tab:red', linestyle='dotted')
    if buyPoints.size > 0:
        plt.plot(pd.to_datetime(buyPoints[: ,0]), buyPoints[: ,1].astype(float),label = "buy", color='black', marker='^', linestyle='', ms=5)
    if sellPoints.size > 0:
        plt.plot(pd.to_datetime(sellPoints[: ,0]), sellPoints[: ,1].astype(float), label = "sell", color='black', marker='v', linestyle='', ms=5)
    if stopPoints.size > 0:
        plt.plot(pd.to_datetime(stopPoints[: ,0]), stopPoints[: ,1].astype(float), label = "stop", color='red', marker='.', linestyle='', ms=5)
    if profitPoints.size > 0:
        plt.plot(pd.to_datetime(profitPoints[: ,0]), profitPoints[: ,1].astype(float), label = "profit", color='yellow', marker='+', linestyle='', ms=5)
    if sellPointsDeniedByEngulfing.size > 0:
        plt.plot(pd.to_datetime(sellPointsDeniedByEngulfing[: ,0]), sellPointsDeniedByEngulfing[: ,1].astype(float), color='black', marker='v', linestyle='', ms=5)
        plt.plot(pd.to_datetime(sellPointsDeniedByEngulfing[: ,0]), sellPointsDeniedByEngulfing[: ,1].astype(float), color='black', marker='x', linestyle='', ms=5)
    if buyPointsDeniedByEngulfing.size > 0:
        plt.plot(pd.to_datetime(buyPointsDeniedByEngulfing[: ,0]), buyPointsDeniedByEngulfing[: ,1].astype(float), color='black', marker='^', linestyle='', ms=5)
        plt.plot(pd.to_datetime(buyPointsDeniedByEngulfing[: ,0]), buyPointsDeniedByEngulfing[: ,1].astype(float),  color='black', marker='x', linestyle='', ms=5)
    title = f'Data: {currentDay} Dia: {day} Trade: R${amountPerDay} ({trade}%) Carteira:  {money}'
    
    
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    plt.title(title)
    plt.legend()
    plt.xlabel("Hora")
    plt.ylabel("Valor do ativo (close)")
    plt.grid()
    
    xValues = []
    for time in times:
        if time.minute == 0:
            xValues.append(time)
    plt.xticks(xValues)
    plt.savefig(f'{pathImgs}/{title}.png')
    plt.close()

def appendPoints(numpyArray, point):
    if(numpyArray.size==0):
        numpyArray = np.append(numpyArray,point)
        numpyArray = numpyArray.reshape((1,len(point)))
    else:
        numpyArray = np.append(numpyArray,[point], axis=0)
    return numpyArray
    
def runStrategy(df, dailyDf, Money, stopLoss,errorMax,errorMin, stdMax, stdMin, alfa, beta, method, ticker, verbose=True, checkEngulfing=False):
    
    datetime.timestamp(datetime.now())
    timestamp = datetime.timestamp(datetime.now())
    pathDebug = f'./{ticker}_debug_{timestamp}'
    if(verbose):
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

        # obtém todos os tempos( x em x min) do dia currentDay
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
        
        buyPointsDeniedByEngulfing = np.array([])
        sellPointsDeniedByEngulfing = np.array([])
        buyPoints = np.array([])
        sellPoints = np.array([])
        stopPoints = np.array([])
        profitPoints = np.array([])

        for i in range(0,daySize):
            if(i==0):
                continue
            currentTime = dayDf.iloc[i]
            previousTime = dayDf.iloc[i-1]
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
                    engulfingCondition = currentTime.open < previousTime.open and currentTime.close > previousTime.close
                    if(checkEngulfing and not(engulfingCondition)):
                        buyPointsDeniedByEngulfing  = appendPoints(buyPointsDeniedByEngulfing,[currentTime.time,currentTime.close])
                        continue
                    buyPoints = appendPoints(buyPoints,[currentTime.time,currentTime.close])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(currentTime.close, Money)
                    opPrice = currentTime.close
                    op = 'buy'
                    trade = [currentTime.date, quantity, currentTime.time, 'Buy', currentTime.close]
                elif(currentTime.close >=  highPred):
                    engulfingCondition = currentTime.open > previousTime.open and currentTime.close < previousTime.close
                    if(checkEngulfing and not(engulfingCondition)):
                        sellPointsDeniedByEngulfing  = appendPoints(sellPointsDeniedByEngulfing,[currentTime.time,currentTime.close])
                        continue
                    sellPoints = appendPoints(sellPoints,[currentTime.time,currentTime.close])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(currentTime.close, Money)
                    opPrice = currentTime.close
                    op = 'sell'
                    trade = [currentTime.date, quantity, currentTime.time, 'Sell', currentTime.close]
        if day%1 == 0 and verbose:
            makeImgDaily(
                currentTime.date, 
                day,
                dayDf,
                buyPoints,
                sellPoints,
                stopPoints,
                profitPoints,
                amountPerDay,
                Money,
                pathDebug,
                sellPointsDeniedByEngulfing= sellPointsDeniedByEngulfing,
                buyPointsDeniedByEngulfing= buyPointsDeniedByEngulfing
                )  
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



def runLuizStrategy(df, dailyDf, Money, stopLoss,errorMax,errorMin, stdMax, stdMin, alfa, beta, method, ticker, verbose=True, checkEngulfing=False):
    
    datetime.timestamp(datetime.now())
    timestamp = datetime.timestamp(datetime.now())
    pathDebug = f'./{ticker}_debug_{timestamp}'
    if(verbose):
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

        # obtém todos os tempos( x em x min) do dia currentDay
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
        
        buyPointsDeniedByEngulfing = np.array([])
        sellPointsDeniedByEngulfing = np.array([])
        buyPoints = np.array([])
        sellPoints = np.array([])
        stopPoints = np.array([])
        profitPoints = np.array([])
        

        for i in range(0,daySize):
            currentTime = dayDf.iloc[i]
            previousTime = dayDf.iloc[i-1]
            if (abs((1-errorMax)*highPred - (1+errorMin)*lowPred )/((1+errorMin)*lowPred) < 0.01):
                continue
            if(i==0):
                continue
            
            
            
            if(countStop >= 30):
                countStop = countStop
                break
            
            if(op != 'none'):
                isLastMoment = i == daySize-1
                
                stopLossPriceWhenOperationIsBought = opPrice*(1-stopLoss)
                stopLossPriceWhenOperationIsSales = opPrice*(1+stopLoss)
                
                buyGoalPrice = (1-errorMax)*highPred
                sellGoalPrice = (1+errorMin)*lowPred
                
                isBuyStop = currentTime.low < stopLossPriceWhenOperationIsBought
                # isBuyStopPrevious = previousTime.low < stopLossPriceWhenOperationIsBought
                isBuyStopPrevious = False
                isProfitBuy = currentTime.high >= buyGoalPrice
                
                isSellStop = currentTime.high > stopLossPriceWhenOperationIsSales
                # isSellStopPrevious = previousTime.high > stopLossPriceWhenOperationIsSales
                isSellStopPrevious = False
                isSellProfit = currentTime.close <= (1+errorMin)*lowPred
                
                if((op == 'buy') and (isBuyStop or isBuyStopPrevious or isProfitBuy or isLastMoment )):
                    if(isBuyStop): 
                        countStop = countStop+1
                        stopPoints = appendPoints(stopPoints,[currentTime.time,stopLossPriceWhenOperationIsBought])
                        realPriceBuy = stopLossPriceWhenOperationIsBought
                    elif(isBuyStopPrevious):
                        countStop = countStop+1
                        stopPoints = appendPoints(stopPoints,[previousTime.time,stopLossPriceWhenOperationIsBought])
                        realPriceBuy = stopLossPriceWhenOperationIsBought
                    elif(isProfitBuy):
                        profitPoints = appendPoints(profitPoints,[currentTime.time,buyGoalPrice])
                        realPriceBuy = buyGoalPrice
                    else:
                        if(currentTime.close >= opPrice):
                            profitPoints = appendPoints(profitPoints,[currentTime.time,currentTime.close])
                        else:
                            stopPoints = appendPoints(stopPoints,[currentTime.time,currentTime.close])
                        realPriceBuy = currentTime.close
                    investedMoney = (opPrice*100*quantity)
                    traderResult =  (realPriceBuy*100*quantity) - (opPrice*100*quantity)
                    Money = Money + investedMoney + traderResult
                    
                    percentualProfit = 100*traderResult/investedMoney
                    #  'profit', 'totalProfit', 'stop','Amount','max','min'
                    trade = trade + [currentTime.time, 'Sell', currentTime.close, currentTime.close-opPrice,percentualProfit, isStop, Money, highPred, lowPred]
                    orders.append(trade)
                    
                    # amountPerDay = amountPerDay + (currentTime.close-opPrice)*100*quantity
                    amountPerDay = amountPerDay + traderResult
                    
                    quantity = 0
                    op = 'none'
                    
                
                elif((op == 'sell') and (isSellStop or isSellStopPrevious or isSellProfit or isLastMoment)):
                    if(isSellStop): 
                        countStop = countStop+1
                        stopPoints = appendPoints(stopPoints,[currentTime.time,stopLossPriceWhenOperationIsSales])
                        realPriceSell = stopLossPriceWhenOperationIsSales
                    elif(isSellStopPrevious):
                        countStop = countStop+1
                        stopPoints = appendPoints(stopPoints,[previousTime.time,stopLossPriceWhenOperationIsSales])
                        realPriceSell = stopLossPriceWhenOperationIsSales
                    elif(isSellProfit):
                        profitPoints = appendPoints(profitPoints,[currentTime.time,sellGoalPrice])
                        realPriceSell = sellGoalPrice
                    else:
                        if(currentTime.close <= opPrice):
                            profitPoints = appendPoints(profitPoints,[currentTime.time,currentTime.close])
                        else:
                            stopPoints = appendPoints(stopPoints,[currentTime.time,currentTime.close])
                        realPriceSell = currentTime.close
                    investedMoney = (opPrice*100*quantity)
                    traderResult =  -1*((realPriceSell*100*quantity) - investedMoney)
                    Money = Money + investedMoney + traderResult 
                    percentualProfit = 100*traderResult/investedMoney
                    trade = trade + [currentTime.time, 'Buy', currentTime.close, opPrice-currentTime.close,percentualProfit, isStop, Money, highPred, lowPred]
                    orders.append(trade)

                    amountPerDay = amountPerDay + traderResult
                    quantity = 0
                    op = 'none'

            elif(op == 'none' and i < daySize-1 ):
                # disc
                enterPurchasePrice = (1 + errorMin)*lowPred
                enterSalesPrice = (1 - errorMax)*highPred
                if(previousTime.low < enterPurchasePrice and currentTime.open > enterPurchasePrice):
                    opPrice = currentTime.open
                    engulfingCondition = currentTime.open < previousTime.open and currentTime.close > previousTime.close
                    if(checkEngulfing and not(engulfingCondition)):
                        buyPointsDeniedByEngulfing  = appendPoints(buyPointsDeniedByEngulfing,[currentTime.time,opPrice])
                        continue
                    buyPoints = appendPoints(buyPoints,[currentTime.time,opPrice])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(opPrice, Money)
                    op = 'buy'
                    trade = [currentTime.date, quantity, currentTime.time, 'Buy',opPrice]
                elif(previousTime.high > enterSalesPrice and currentTime.open < enterSalesPrice):
                    opPrice = currentTime.open
                    engulfingCondition = currentTime.open > previousTime.open and currentTime.close < previousTime.close
                    if(checkEngulfing and not(engulfingCondition)):
                        sellPointsDeniedByEngulfing  = appendPoints(sellPointsDeniedByEngulfing,[currentTime.time,opPrice])
                        continue
                    sellPoints = appendPoints(sellPoints,[currentTime.time,opPrice])
                    tradesPerDay = tradesPerDay + 1
                    Money, quantity = buy(opPrice, Money)
                    op = 'sell'
                    trade = [currentTime.date, quantity, currentTime.time, 'Sell', opPrice]
        if verbose:
            # pass
            makeImgDaily(
                currentTime.date, 
                day,
                dayDf,
                buyPoints,
                sellPoints,
                stopPoints,
                profitPoints,
                amountPerDay,
                Money,
                pathDebug,
                sellPointsDeniedByEngulfing= sellPointsDeniedByEngulfing,
                buyPointsDeniedByEngulfing= buyPointsDeniedByEngulfing,
                maxError=errorMin,
                minError=errorMax
                )  
        # if currentTime:
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

