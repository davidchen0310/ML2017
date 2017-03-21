#!/bin/python3.6
import numpy as np
import pandas as pd
import math
import sys
# import matplotlib.pyplot as plt
# import time


class Parameters:
    pass


# start_time = time.clock()
# print(start_time)

# global variables
no_categories = 18  # number of categories of provided data
index_PM25 = 9  # start from 0
index_PM10 = 8
no_hours = 9  # Use first nine hours to predict the tenth hour
start_hour = [0,0] # index from 0
no_features = np.sum((np.full(len(start_hour), no_hours) - start_hour)) + 1
choose_pm10 = False
train_test_data = False
random = True
no_train_data_elements = 1800
pre_select = True
pre_selection = [1,3,5,10,12,15,27,28,30,34,39,41,42,44,48,49,54,55,57,66,71,77,78,79,80,81,83,95,96,102,103,104,115,128,141,143,145,155,158,159,161,164,166,170,172,177,179,183,185,186,187,192,196,197,198,207,209,218,219,223,227,233,235,236,237,238,251,254,255,257,260,261,265,266,270,278,279,286,288,294,298,299,301,308,327,328,329,330,331,332,337,339,348,349,350,351,356,360,364,367,369,370,377,378,381,384,388,396,402,403,408,409,415,416,419,428,432,434,445,446,449,452,453,454,460,461,462,471,473,475,482,487,489,490,494,496,499,506,507,515,517,518,519,523,524,528,534,541,551,555,556,560,561,565,566,567,568,569,570,573,574,576,580,584,586,589,594,596,598,604,606,612,614,615,616,617,618,630,634,637,639,640,641,646,650,654,657,660,663,674,678,680,686,687,688,690,696,700,702,711,712,716,718,720,724,725,740,741,742,743,745,751,753,759,764,765,767,768,771,772,774,776,779,781,785,789,794,798,803,804,806,809,810,815,816,819,826,828,831,841,843,845,848,850,851,853,860,863,871,879,885,887,888,891,893,895,900,902,903,906,907,910,912,914,915,919,923,925,930,931,934,940,946,950,953,954,959,961,962,970,971,974,975,986,989,991,994,1002,1010,1012,1013,1025,1026,1028,1031,1041,1047,1051,1055,1058,1059,1063,1070,1071,1078,1083,1085,1090,1091,1095,1099,1111,1114,1116,1118,1119,1122,1126,1127,1129,1131,1134,1144,1145,1148,1149,1157,1161,1162,1166,1174,1175,1179,1180,1181,1184,1188,1194,1196,1202,1204,1207,1210,1211,1213,1219,1220,1222,1225,1226,1235,1236,1241,1242,1245,1246,1251,1252,1253,1263,1264,1267,1268,1269,1270,1271,1272,1275,1279,1280,1282,1285,1286,1287,1292,1295,1299,1303,1305,1308,1311,1321,1323,1325,1328,1329,1341,1342,1346,1348,1350,1351,1353,1357,1365,1366,1368,1378,1379,1383,1395,1400,1401,1402,1405,1406,1407,1408,1413,1414,1417,1422,1424,1426,1427,1430,1432,1434,1440,1446,1449,1450,1451,1456,1462,1463,1464,1465,1471,1475,1478,1483,1486,1488,1491,1495,1496,1497,1498,1500,1504,1508,1509,1512,1515,1516,1525,1526,1527,1530,1532,1534,1536,1542,1545,1546,1548,1549,1551,1554,1555,1564,1568,1569,1570,1573,1575,1576,1581,1588,1591,1592,1593,1605,1606,1607,1613,1617,1620,1621,1623,1625,1638,1647,1648,1651,1654,1656,1658,1659,1660,1662,1669,1675,1678,1680,1681,1688,1689,1694,1695,1698,1701,1704,1706,1715,1716,1718,1722,1729,1732,1734,1735,1738,1743,1746,1748,1749,1751,1757,1758,1760,1761,1762,1763,1764,1772,1775,1779,1780,1785,1789,1794,1797,1798,1799,1800,1805,1815,1819,1827,1830,1831,1833,1840,1843,1844,1845,1852,1853,1855,1856,1858,1859,1861,1865,1867,1869,1870,1873,1879,1880,1886,1889,1890,1892,1895,1903,1905,1906,1908,1912,1919,1926,1927,1931,1932,1937,1940,1941,1942,1945,1947,1949,1951,1953,1957,1959,1960,1963,1964,1965,1968,1974,1975,1976,1980,1982,1983,1986,1987,1992,1993,1994,1997,1998,1999,2002,2006,2009,2015,2023,2025,2026,2027,2029,2030,2031,2034,2036,2038,2039,2041,2044,2047,2049,2052,2057,2060,2061,2063,2072,2074,2078,2080,2083,2084,2092,2093,2095,2097,2099,2103,2104,2105,2106,2109,2113,2116,2120,2126,2128,2129,2133,2136,2138,2141,2143,2146,2147,2149,2153,2157,2159,2163,2164,2165,2167,2178,2179,2181,2183,2187,2191,2195,2202,2203,2204,2208,2209,2210,2211,2225,2227,2228,2229,2232,2240,2242,2244,2248,2249,2250,2252,2253,2254,2255,2258,2268,2269,2270,2271,2278,2279,2281,2282,2284,2287,2293,2295,2298,2300,2305,2307,2314,2316,2320,2328,2329,2331,2335,2337,2338,2340,2343,2346,2347,2348,2350,2351,2352,2353,2357,2358,2359,2361,2366,2367,2370,2372,2374,2378,2379,2380,2383,2389,2391,2393,2394,2397,2398,2400,2404,2405,2406,2407,2410,2412,2413,2415,2417,2420,2421,2426,2427,2429,2430,2432,2438,2440,2442,2448,2452,2453,2461,2465,2467,2470,2471,2472,2475,2480,2482,2495,2497,2503,2504,2508,2511,2513,2519,2522,2523,2527,2528,2533,2534,2537,2539,2540,2544,2546,2556,2560,2563,2564,2565,2571,2580,2586,2591,2594,2597,2599,2600,2606,2615,2616,2617,2619,2624,2627,2628,2631,2632,2633,2638,2641,2644,2645,2649,2654,2655,2657,2661,2662,2663,2664,2669,2670,2672,2681,2683,2684,2685,2688,2692,2698,2701,2705,2706,2707,2712,2714,2718,2719,2720,2722,2725,2726,2727,2730,2733,2736,2738,2742,2746,2747,2751,2758,2760,2761,2767,2769,2771,2774,2775,2776,2779,2780,2781,2794,2795,2796,2799,2801,2803,2804,2807,2808,2809,2810,2812,2814,2815,2817,2818,2824,2826,2829,2831,2832,2833,2834,2835,2836,2840,2841,2844,2845,2847,2848,2865,2866,2867,2872,2875,2877,2881,2882,2885,2892,2893,2894,2895,2896,2898,2899,2908,2912,2914,2915,2918,2924,2926,2928,2929,2935,2938,2940,2942,2944,2946,2948,2949,2952,2953,2955,2957,2958,2959,2965,2968,2969,2972,2977,2979,2981,2987,2995,2996,2998,3001,3002,3006,3010,3015,3019,3021,3024,3028,3030,3039,3041,3045,3046,3047,3049,3051,3053,3054,3055,3063,3066,3067,3072,3079,3082,3086,3090,3091,3093,3097,3102,3104,3110,3113,3118,3123,3125,3126,3128,3130,3134,3137,3139,3140,3141,3142,3145,3156,3158,3159,3162,3164,3168,3172,3176,3178,3181,3183,3184,3186,3193,3194,3197,3200,3204,3217,3222,3224,3228,3229,3230,3232,3233,3236,3237,3240,3241,3242,3245,3247,3248,3249,3252,3257,3260,3262,3269,3273,3278,3285,3286,3287,3288,3289,3290,3292,3299,3300,3302,3303,3310,3316,3318,3319,3320,3321,3325,3328,3331,3335,3339,3342,3344,3345,3346,3347,3351,3352,3354,3361,3362,3364,3375,3382,3386,3389,3390,3392,3393,3397,3401,3405,3410,3411,3412,3413,3416,3417,3418,3421,3423,3427,3431,3432,3440,3448,3450,3451,3456,3457,3465,3469,3471,3472,3476,3479,3480,3481,3487,3493,3496,3505,3510,3511,3517,3521,3522,3526,3528,3529,3541,3543,3547,3549,3550,3553,3555,3559,3565,3566,3579,3581,3585,3587,3589,3590,3596,3598,3601,3603,3608,3609,3616,3619,3620,3625,3629,3631,3633,3635,3644,3651,3653,3654,3664,3667,3668,3672,3679,3681,3682,3683,3685,3690,3693,3696,3697,3712,3717,3718,3722,3729,3731,3732,3733,3741,3742,3743,3744,3747,3750,3757,3759,3765,3766,3775,3778,3784,3785,3791,3800,3801,3812,3814,3816,3817,3819,3822,3825,3826,3830,3832,3848,3853,3855,3857,3858,3859,3861,3863,3872,3873,3874,3879,3881,3885,3886,3889,3895,3899,3902,3908,3913,3914,3919,3920,3924,3928,3934,3935,3938,3941,3942,3945,3946,3947,3948,3952,3954,3959,3960,3961,3963,3966,3967,3968,3970,3972,3973,3985,3986,3987,3991,3992,3993,3994,3996,3998,3999,4001,4002,4004,4010,4012,4015,4026,4027,4029,4033,4035,4038,4048,4049,4052,4055,4061,4062,4066,4067,4068,4070,4073,4074,4077,4079,4082,4083,4087,4088,4094,4095,4098,4103,4104,4106,4109,4114,4116,4117,4119,4122,4124,4125,4129,4131,4134,4136,4137,4140,4141,4143,4147,4149,4152,4156,4157,4158,4170,4174,4176,4178,4181,4182,4187,4188,4191,4194,4195,4196,4197,4199,4202,4207,4209,4211,4213,4219,4223,4224,4225,4226,4229,4231,4232,4233,4237,4241,4242,4243,4249,4254,4255,4257,4258,4259,4263,4265,4266,4268,4270,4271,4274,4276,4277,4280,4281,4282,4286,4288,4289,4294,4295,4297,4307,4312,4315,4316,4317,4318,4319,4321,4322,4329,4342,4346,4349,4351,4354,4356,4359,4369,4373,4375,4376,4378,4380,4382,4389,4390,4392,4396,4398,4400,4401,4402,4409,4410,4411,4413,4414,4421,4422,4423,4425,4431,4436,4437,4439,4440,4441,4442,4451,4453,4459,4461,4463,4465,4470,4474,4475,4478,4486,4488,4492,4494,4495,4500,4505,4510,4511,4517,4525,4528,4530,4538,4539,4540,4541,4547,4551,4554,4557,4559,4569,4573,4575,4589,4590,4598,4601,4603,4610,4622,4628,4629,4631,4632,4634,4637,4640,4644,4645,4648,4649,4650,4653,4654,4657,4659,4664,4665,4671,4673,4674,4677,4679,4680,4682,4685,4687,4691,4695,4696,4698,4701,4704,4706,4712,4714,4715,4718,4721,4734,4737,4739,4748,4749,4750,4752,4759,4761,4765,4769,4770,4771,4773,4774,4777,4778,4782,4785,4786,4787,4791,4792,4795,4797,4800,4801,4803,4811,4812,4813,4821,4826,4829,4837,4839,4844,4845,4852,4857,4859,4862,4863,4866,4870,4875,4876,4878,4879,4883,4884,4888,4890,4893,4896,4897,4898,4900,4901,4906,4908,4910,4917,4918,4921,4927,4929,4930,4932,4936,4937,4938,4939,4940,4941,4942,4943,4944,4948,4952,4961,4968,4969,4970,4975,4976,4978,4981,4983,4993,4994,4997,4998,5007,5011,5012,5014,5026,5031,5033,5037,5039,5041,5044,5049,5050,5055,5060,5063,5064,5070,5071,5073,5075,5077,5081,5082,5083,5090,5097,5103,5104,5105,5107,5108,5113,5123,5131,5134,5137,5141,5143,5144,5147,5148,5151,5153,5155,5158,5159,5164,5170,5172,5173,5178,5183,5187,5189,5194,5199,5200,5202,5203,5204,5205,5207,5214,5219,5220,5222,5223,5226,5227,5228,5234,5237,5238,5240,5243,5247,5248,5252,5257,5262,5265,5271,5272,5276,5281,5284,5294,5296,5297,5299,5306,5307,5318,5321,5323,5324,5328,5332,5336,5339,5341,5344,5349,5350,5351,5352,5353,5354,5356,5359,5362,5363,5371,5373,5375,5376,5381,5383,5391,5394,5396,5401,5405,5408,5410,5412,5416,5418,5423,5425,5429,5431,5435,5436,5442,5446,5449,5452,5456,5458,5460,5464,5469,5472,5473,5476,5478,5480,5489,5496,5503,5507,5508,5509,5513,5514,5515,5517,5518,5522,5526,5529,5530,5532,5537,5545,5550,5551,5556,5557,5558,5560,5561,5563,5574,5575,5581,5604,5607,5617,5620,5625,5629,5631,5633,5636,5643,5647,5648,5650,5651]

training_file = sys.argv[1]
test_file = sys.argv[2]
res_file = sys.argv[3]
parameters_file = "parameters.txt"
seeds_file = "seeds.txt"
err_file = "errors.txt"
loss_file = "loss.txt"


def get_training_data():

    matrices = pd.read_csv(training_file,
                           encoding="big5",
                           usecols=range(3, 27),
                           chunksize=no_categories)

    # 12 months, 20 days, 24 hours
    raw_data = np.empty((len(start_hour), 12 * 20 * 24))

    if len(start_hour) is 1 or len(start_hour) is 2:
        if len(start_hour) is 2 and choose_pm10 is True:
            for i, matrix in enumerate(matrices):
                for j, row in enumerate(matrix.values):
                    if j is index_PM25:
                        raw_data[0][i * 24:(i + 1) * 24] = row  
                    if j is index_PM10:
                        raw_data[1][i * 24:(i + 1) * 24] = row
        else:
            for i, matrix in enumerate(matrices):
                for j, row in enumerate(matrix.values):
                    if j is index_PM25:
                        raw_data[0][i * 24:(i + 1) * 24] = row    
            if len(start_hour) is 2:
                raw_data[1] = raw_data[0] ** 2
    
    if len(start_hour) is 3 or len(start_hour) is 4:
        for i, matrix in enumerate(matrices):
            for j, row in enumerate(matrix.values):
                if j is index_PM25:
                    raw_data[0][i * 24:(i + 1) * 24] = row  
                if j is index_PM10:
                    raw_data[2][i * 24:(i + 1) * 24] = row

        raw_data[1] = raw_data[0] ** 2
        if len(start_hour) is 4:
            raw_data[3] = raw_data[2] ** 2

    no_elements = 12 * (20 * 24 - no_hours)

    data = np.empty((no_elements, no_features))
    target = np.empty(no_elements)

    for m in range(12):  # months
        for i in range(20 * 24 - no_hours):
            index = math.floor(m * no_elements / 12 + i)
            offset = 480 * m + i
            data[index][0] = 1
            data[index][1:] = np.hstack([r[offset + start_hour[j]:offset + no_hours] for j, r in enumerate(raw_data)])
            target[index] = raw_data[0][offset + no_hours]

    temp_data = np.empty((no_train_data_elements, no_features))
    temp_target = np.empty(no_train_data_elements)
    selection = np.array(sorted(np.random.choice(no_elements, size=no_train_data_elements, replace=False)))
    if pre_select is True:
        selection = pre_selection
    select_index = 0

    if random is True:
        for i, (d, t) in enumerate(zip(data, target)):
            if i == selection[select_index]:
                temp_data[select_index] = d
                temp_target[select_index] = t
                select_index = select_index + 1
                if select_index == no_train_data_elements:
                    break;
        # print(selection[0])
        # print(data[selection[0]])
        # print(temp_data[0])
        data = temp_data
        target = temp_target

    # print(data.shape)
    # print(target.shape)

    if train_test_data is True:

        matrices = pd.read_csv(test_file,
                               usecols=range(2, 11),
                               header=None,
                               chunksize=no_categories)

        raw_test_data = np.empty((len(start_hour), 9 * 240))

        if len(start_hour) is 1 or len(start_hour) is 2:
            if len(start_hour) is 2 and choose_pm10 is True:
                for i, matrix in enumerate(matrices):
                    for j, row in enumerate(matrix.values):
                        if j is index_PM25:
                            raw_test_data[0][i * 9:(i + 1) * 9] = row  
                        if j is index_PM10:
                            raw_test_data[1][i * 9:(i + 1) * 9] = row
            else:
                for i, matrix in enumerate(matrices):
                    for j, row in enumerate(matrix.values):
                        if j is index_PM25:
                            raw_test_data[0][i * 9:(i + 1) * 9] = row    
                if len(start_hour) is 2:
                    raw_test_data[1] = raw_test_data[0] ** 2
        
        if len(start_hour) is 3 or len(start_hour) is 4:
            for i, matrix in enumerate(matrices):
                for j, row in enumerate(matrix.values):
                    if j is index_PM25:
                        raw_test_data[0][i * 9:(i + 1) * 9] = row  
                    if j is index_PM10:
                        raw_test_data[2][i * 9:(i + 1) * 9] = row

            raw_test_data[1] = raw_test_data[0] ** 2
            if len(start_hour) is 4:
                raw_test_data[3] = raw_test_data[2] ** 2

        elements_per_row = min(start_hour)

        no_elements = 240 * elements_per_row

        test_data = np.empty((no_elements, no_features))
        test_target = np.empty(no_elements)

        for e in range(240):  # 240 raw test elements (240 rows)
            for i in range(elements_per_row):
                index = e * elements_per_row + i
                offset = e * 9 + i - elements_per_row
                test_data[index][0] = 1
                test_data[index][1:] = np.hstack([r[offset + start_hour[j]:offset + no_hours] for j, r in enumerate(raw_test_data)])
                test_target[index] = raw_test_data[0][offset + no_hours]

        return(np.concatenate((data, test_data), axis=0), np.concatenate((target, test_target), axis=0))

    return (selection, data, target)


def train(data, target):

    p = Parameters()

    # p.w = np.full(len(data[0]), 0.1).flatten() 
    # p.w[0][0] = 0
    # p._lambda = 0    

    # p.w = np.random.rand(len(data[0])).flatten()  # initialize weights
    # p._lambda = 0  # regularization

    p.w = np.zeros(len(data[0])).flatten() 
    p._lambda = 0

    lr = np.ones(len(data[0])).flatten()  # learning rate
    iteration = 300000

    grad_sum = np.zeros(len(data[0])).flatten()

    # err_over_iterations = np.empty(iteration)

    # Iterations
    for i in range(iteration):

        prediction = np.dot(data, p.w)

        grad = 2 * np.dot(data.T, prediction - target) + p._lambda * np.sum(p.w)

        # Update parameters
        grad_sum = grad_sum + grad ** 2
        ada = np.sqrt(grad_sum)
        p.w = p.w - lr / ada * grad

        # print(prediction)
        # print(target)

        # err_over_iterations[i] = rmse(prediction, target)

    # print(err_over_iterations)
    # save_loss(err_over_iterations)
    # plt.plot(err_over_iterations)

    return p


def save_loss(errs):
    with open(loss_file, "wb") as f:
        for i, err in enumerate(errs):
            np.savetxt(f, np.array([i, err]).reshape((1, 2)), fmt="%1.7f")


def get_test_data():
    matrices = pd.read_csv(test_file,
                           usecols=range(2, 11),
                           header=None,
                           chunksize=no_categories)

    actual_hours = np.full(len(start_hour), no_hours) - start_hour

    # 12 months, 20 days, 24 hours
    test_data = np.empty((240, no_features))

    if len(start_hour) is 1:
        for i, matrix in enumerate(matrices):
            test_data[i][0] = 1
            for j, row in enumerate(matrix.values):
                if j is index_PM25:
                    test_data[i][1:actual_hours[0] + 1] = row[start_hour[0]:]

    if len(start_hour) is 2:
        if choose_pm10 is True:
            for i, matrix in enumerate(matrices):
                test_data[i][0] = 1
                for j, row in enumerate(matrix.values):
                    if j is index_PM25:
                        test_data[i][1:actual_hours[0] + 1] = row[start_hour[0]:]
                    if j is index_PM10:
                        test_data[i][actual_hours[0] + 1:np.sum(actual_hours[:2]) + 1] = row[start_hour[1]:]
        else:   
            for i, matrix in enumerate(matrices):
                test_data[i][0] = 1
                for j, row in enumerate(matrix.values):
                    if j is index_PM25:
                        test_data[i][1:actual_hours[0] + 1] = row[start_hour[0]:]
                        test_data[i][actual_hours[0] + 1:np.sum(actual_hours[:2]) + 1] = np.array(row[start_hour[1]:], dtype=float) ** 2

    if len(start_hour) is 3:
        for i, matrix in enumerate(matrices):
            test_data[i][0] = 1
            for j, row in enumerate(matrix.values):
                if j is index_PM25:
                    test_data[i][1:actual_hours[0] + 1] = row[start_hour[0]:]
                    test_data[i][actual_hours[0] + 1:np.sum(actual_hours[:2]) + 1] = np.array(row[start_hour[1]:], dtype=float) ** 2
                if j is index_PM10:
                    test_data[i][np.sum(actual_hours[:2]) + 1:np.sum(actual_hours[:3]) + 1] = row[start_hour[2]:]

    if len(start_hour) is 4:
        for i, matrix in enumerate(matrices):
            test_data[i][0] = 1
            for j, row in enumerate(matrix.values):
                if j is index_PM25:
                    test_data[i][1:actual_hours[0] + 1] = row[start_hour[0]:]
                    test_data[i][actual_hours[0] + 1:np.sum(actual_hours[:2]) + 1] = np.array(row[start_hour[1]:], dtype=float) ** 2
                if j is index_PM10:
                    test_data[i][np.sum(actual_hours[:2]) + 1:np.sum(actual_hours[:3]) + 1] = row[start_hour[2]:]
                    test_data[i][np.sum(actual_hours[:3]) + 1:np.sum(actual_hours[:4]) + 1] = np.array(row[start_hour[3]:], dtype=float) ** 2

    return test_data


def save_predictions(predictions):
    with open(res_file, "w") as f:
        title = "id" + "," + "value" + '\n'
        f.write(title)
        for i, prediction in enumerate(np.squeeze(predictions)):
            row = "id_" + str(i) + "," + str(prediction) + '\n'
            f.write(row)


def get_targets():
    return np.loadtxt("correct-answer.csv",
                      usecols=[1],
                      delimiter=',')


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def save_seeds(seeds):
    with open(seeds_file, "wb") as f:
        for i, s in enumerate(seeds):
            np.savetxt(f, to_arr("seed_id_" + str(i)), fmt="%s")
            np.savetxt(f, np.expand_dims(s, axis=0), delimiter=',', fmt='%d')


# def save_parameters(p_arr):
#     with open(parameters_file, "wb") as f:
#         for i, p in enumerate(p_arr):
#             np.savetxt(f, to_arr("seed_id_" + str(i)), fmt="%s")
#             np.savetxt(f, p.w, delimiter=',', fmt='%1.7f')
#             np.savetxt(f, to_arr(p._lambda), fmt='%d')


def save_errs(train_arr, test_arr):
    with open(err_file, "wb") as f:
        for i, (train, test) in enumerate(zip(train_arr, test_arr)):
            np.savetxt(f, to_arr(i), newline=' ', fmt='%d')
            np.savetxt(f, to_arr(train), newline=' ', fmt='%1.7f')
            np.savetxt(f, to_arr(test), fmt='%1.7f')


def to_arr(output):
    return np.array(output).reshape((1))


# def plot_scatter():
#     x = get_training_data_row(index_PM25)

#     for i in range(10, 18):
#         if i != index_PM25 and i != 10:
#             y = get_training_data_row(i)
#             plt.subplot(4, 2, i - 9)
#             plt.scatter(x, y)

#     plt.show()


no_seed = 1
train_err_arr = np.empty(no_seed)
test_err_arr = np.empty(no_seed)
seeds = np.empty((no_seed, no_train_data_elements))

for seed_index in range(no_seed):

    # no_train_data_elements = seed_index * 100

    # plt.figure(seed_index + 1)
        
    # no_features = np.sum((np.full(len(start_hour), no_hours) - start_hour)) + 1

    seed, data, target = get_training_data()
    seeds[seed_index] = seed

    parameters = train(data, target)
    # train_err_arr[seed_index] = rmse(np.dot(data, parameters.w), target)

    # print(i+1, j+1, train_err, sep=' ')

    predictions = np.dot(get_test_data(), parameters.w)
    save_predictions(predictions)
    # test_err_arr[seed_index] = rmse(predictions, get_targets())

    # print(i+1, j+1, train_err, test_err_arr[seed_index], sep=' ')

        # data, target = get_training_data()
        # parameters, train_err = train(data, target)
        # train_err_arr[seed_index] = train_err
        # parameters_arr.append(parameters)

        # predictions = np.dot(get_test_data(), parameters.w).flatten()

        # save_predictions(predictions)
        # test_err_arr[seed_index] = rmse(predictions , get_targets())

        # for i, (p, t) in enumerate(zip(predictions, get_targets())):
        #     print(i, abs(p - t), sep=' ')

# print(sorted(train_err_arr))
# print(sorted(test_err_arr))

# save_seeds(seeds)
# save_errs(train_err_arr, test_err_arr)

# end_time = time.clock()
# print(end_time - start_time)

# plt.show()
