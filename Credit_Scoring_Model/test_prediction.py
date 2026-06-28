from src.predict import predict_credit_score

sample_customer = {
    "RevolvingUtilizationOfUnsecuredLines": 0.45,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.32,
    "MonthlyIncome": 5500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
}

result, confidence = predict_credit_score(sample_customer)

print("Prediction :", result)
print(f"Confidence : {confidence:.2%}")