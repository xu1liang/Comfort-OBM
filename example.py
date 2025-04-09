#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations
from sim.api import *
from train.xgboost_model import *
import time
from bokeh.plotting import figure, output_file, show, row, column
from bokeh.models import Panel, Tabs
import random
from scipy.stats import norm
def Derivative(array, time_interval):
    differences = np.diff(array)
    first_derivative = differences / time_interval
    return first_derivative

def plot_good_and_hard(goodBehaviorTaskId, badBehaviorTaskId):
    # run_feature_split_test决定是否开启不同的feature组合测试，feature_split_num为拆分的特征数量
    # run_feature_split_test = False
    # feature_split_num = 2
    # features = ['ego_lat_jerk',
    #     'ego_dw',
    #     'ego_w',
    #     'ego_lat_acc',
    #     'ego_steer_angle_rate',
    #     'ego_v'
    #     ]
    # if run_feature_split_test:
    #     feature_pairs = [list(pair) for pair in combinations(features, feature_split_num)]
    # else:
    # feature_pairs = [features]


    checker_name = 'dd_sudden_turn'
    feature_list, name_list = ExtractFeatureMetrics([goodBehaviorTaskId, badBehaviorTaskId], checker_name)


    print("keys: ", (feature_list[0][0]).keys())

    # one dimension plot
    good_lat_jerk = sum([feature["ego_lat_jerk"] for feature in feature_list[0]], [])
    bad_lat_jerk = sum([feature["ego_lat_jerk"] for feature in feature_list[1]], [])
    good_max_lat_jerk = [max(feature["ego_lat_jerk"], key=abs) for feature in feature_list[0]]
    bad_max_lat_jerk = [max(feature["ego_lat_jerk"], key=abs) for feature in feature_list[1]]
    # sum(x, [])将二维列表展平为一维列表
    good_ego_steer_angle = sum([feature["ego_steer_angle"] for feature in feature_list[0]], [])
    bad_ego_steer_angle = sum([feature["ego_steer_angle"] for feature in feature_list[1]], [])

    good_ego_lat_acc = sum([feature["ego_lat_acc"] for feature in feature_list[0]], [])
    bad_ego_lat_acc = sum([feature["ego_lat_acc"] for feature in feature_list[1]], [])
    good_max_lat_acc = [max(feature["ego_lat_acc"], key=abs) for feature in feature_list[0]]
    bad_max_lat_acc = [max(feature["ego_lat_acc"], key=abs) for feature in feature_list[1]]

    good_ego_steer_angle_rate = sum([feature["ego_steer_angle_rate"] for feature in feature_list[0]], [])
    bad_ego_steer_angle_rate = sum([feature["ego_steer_angle_rate"] for feature in feature_list[1]], [])

    planning_time_gap = 0.1
    good_dego_steer_angle = Derivative(good_ego_steer_angle, planning_time_gap)
    bad_dego_steer_angle = Derivative(bad_ego_steer_angle, planning_time_gap)
    good_ddego_steer_angle = Derivative(good_dego_steer_angle, planning_time_gap)
    bad_ddego_steer_angle = Derivative(bad_dego_steer_angle, planning_time_gap)



    def _plot_one_dimension(good_x, good_y, bad_x, bad_y, title):
        p= figure(title=title, x_axis_label='X-Axis', y_axis_label='Y-Axis', plot_width=1920, plot_height=300, y_range=(-0.5, 0.5))
        p.scatter(good_x, good_y, size=5, marker='x',color="red", fill_alpha=0.2, alpha=0.8, legend_label = "good behavior")
        p.circle_x(bad_x, bad_y, size=5, marker='o', color="blue", fill_alpha=0.2, alpha=0.8, legend_label = "bad behavior")
        p.legend.click_policy = "hide"
        p.legend.location = "top_left"
        return p

    def _cal_gauss(x):
        hist, edges = np.histogram(x, density=True, bins=50)
        mu, std = norm.fit(x)
        x = np.linspace(min(x), max(x), 1000)
        pdf = norm.pdf(x, mu, std)
        return hist, edges, x, pdf

    def _plot_good_and_bad_gauss(good_x, bad_x, title):
        good_hist, good_edges, good_x, good_pdf = _cal_gauss(good_x)
        p = figure(title=title, x_axis_label='Value', plot_width=800, plot_height=300, y_axis_label='Probability Density')
        p.quad(top=good_hist, bottom=0, left=good_edges[:-1], right=good_edges[1:], fill_color="red", line_color="white", alpha=0.5, legend_label="Good Histogram")
        p.line(good_x, good_pdf, line_width=2, color="red", legend_label="Good Gaussian")
        bad_hist, bad_edges, bad_x, bad_pdf = _cal_gauss(bad_x)
        p.quad(top=bad_hist, bottom=0, left=bad_edges[:-1], right=bad_edges[1:], fill_color="blue", line_color="white", alpha=0.5, legend_label="Bad Histogram")
        p.line(bad_x, bad_pdf, line_width=2, color="blue", legend_label="Bad Gaussian")
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        return p

    p1 = _plot_one_dimension(good_ego_lat_acc, [0] * len(good_ego_lat_acc), bad_ego_lat_acc, [0] * len(bad_ego_lat_acc), "LatAcc")
    p1_gauss = _plot_good_and_bad_gauss(good_ego_lat_acc, bad_ego_lat_acc, "LatAcc-Gauss")
    p2 = _plot_one_dimension(good_max_lat_acc, [0] * len(good_max_lat_acc), bad_max_lat_acc, [0] * len(bad_max_lat_acc), "MaxLatAcc")
    p2_gauss = _plot_good_and_bad_gauss(good_max_lat_acc, bad_max_lat_acc, "MaxLatAcc-Gauss")
    p3 = _plot_one_dimension(good_lat_jerk, [0] * len(good_lat_jerk), bad_lat_jerk, [0] * len(bad_lat_jerk), "LatJerk")
    p3_gauss = _plot_good_and_bad_gauss(good_lat_jerk, bad_lat_jerk, "LatJerk-Gauss")
    p4 = _plot_one_dimension(good_max_lat_jerk, [0] * len(good_max_lat_jerk), bad_max_lat_jerk, [0] * len(bad_max_lat_jerk), "MaxLatJerk")
    p4_gauss = _plot_good_and_bad_gauss(good_max_lat_jerk, bad_max_lat_jerk, "MaxLatJerk-Gauss")

    p5 = _plot_one_dimension(good_ego_steer_angle, [0] * len(good_ego_steer_angle), bad_ego_steer_angle, [0] * len(bad_ego_steer_angle), "SteerAngle")
    p5_gauss = _plot_good_and_bad_gauss(good_ego_steer_angle, bad_ego_steer_angle, "SteerAngle-Gauss")
    p6 = _plot_one_dimension(good_ego_steer_angle_rate, [0] * len(good_ego_steer_angle_rate), bad_ego_steer_angle_rate, [0] * len(bad_ego_steer_angle_rate), "SteerAngleRate")
    p6_gauss = _plot_good_and_bad_gauss(good_ego_steer_angle_rate, bad_ego_steer_angle_rate, "SteerAngleRate-Gauss")
    p7 = _plot_one_dimension(good_dego_steer_angle, [0] * len(good_dego_steer_angle), bad_dego_steer_angle, [0] * len(bad_dego_steer_angle), "DSteerAngle")
    p7_gauss = _plot_good_and_bad_gauss(good_dego_steer_angle, bad_dego_steer_angle, "DSteerAngle-Gauss")
    p8 = _plot_one_dimension(good_ddego_steer_angle, [0] * len(good_ddego_steer_angle), bad_ddego_steer_angle, [0] * len(bad_ddego_steer_angle), "DDSteerAngle")
    p8_gauss = _plot_good_and_bad_gauss(good_ddego_steer_angle, bad_ddego_steer_angle, "DDSteerAngle-Gauss")

    layout = row(column(p1, p2, p3, p4, p5, p6, p7, p8), column(p1_gauss, p2_gauss, p3_gauss, p4_gauss, p5_gauss, p6_gauss, p7_gauss, p8_gauss))
    return layout

if __name__ == "__main__":
    start_time = time.time()
    html_path = "/Users/liangxu/PycharmProjects/pythonProject/fig/comfort_obm.html"
    output_file(html_path, mode = "inline", title="Comfort OBM")
    task_map = {
        "高速变道晃动Hard": "67f5e29f02daf0bcf4461bc1",
        "高速变道晃动Good": "67f5e29fd7c5690104005ee5",
        "高速非变道晃动Hard": "67f5e29ed7c5690104005eb7",
        "高速非变道晃动Good": "67f5e29ed7c5690104005eb6",
        "城区变道晃动Hard": "67f5e29e02daf0bcf4461bbb",
        "城区变道晃动Good": "67f5e29ed7c5690104005ed1",
        "城区非变道晃动Hard": "67f5e29ea236be4bdef709d3",
        "城区非变道晃动Good": "67f5e2a0a236be4bdef70a1c",
        "猛打场景库Hard": "67f5e29f02daf0bcf4461bd1",
        "猛打场景库Good": "67f5e29f02daf0bcf4461bf7",
        "画龙Hard": "67f5e29ed7c5690104005eba",
        "画龙Good": "67f5e29ed7c5690104005ec2"
    }

    # 晃动FIT
    shake_hnp_lc_layout = plot_good_and_hard(task_map["高速变道晃动Good"], task_map["高速变道晃动Hard"])
    shake_hnp_lc_tab = Panel(child = shake_hnp_lc_layout, title = "晃动FIT-高速变道")

    shake_hnp_lk_layout = plot_good_and_hard(task_map["高速非变道晃动Good"], task_map["高速非变道晃动Hard"])
    shake_hnp_lk_tab = Panel(child = shake_hnp_lk_layout, title = "晃动FIT-高速非变道")

    shake_unp_lc_layout = plot_good_and_hard(task_map["城区变道晃动Good"], task_map["城区变道晃动Hard"])
    shake_unp_lc_tab = Panel(child = shake_unp_lc_layout, title = "晃动FIT-城区变道")
    shake_unp_lk_layout = plot_good_and_hard(task_map["城区非变道晃动Good"], task_map["城区非变道晃动Hard"])
    shake_unp_lk_tab = Panel(child = shake_unp_lk_layout, title = "晃动FIT-城区非变道")

    # 猛打FIT
    sudden_turn_layout = plot_good_and_hard(task_map["猛打场景库Good"], task_map["猛打场景库Hard"])
    sudden_turn_tab = Panel(child = sudden_turn_layout, title = "猛打FIT")

    # 画龙FIT
    draw_layout = plot_good_and_hard(task_map["画龙Good"], task_map["画龙Hard"])
    draw_tab = Panel(child = draw_layout, title = "画龙FIT")

    show(Tabs(tabs=[shake_hnp_lc_tab, shake_hnp_lk_tab, shake_unp_lc_tab, shake_unp_lk_tab, sudden_turn_tab, draw_tab]))

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")


    # print((feature_list[0][0]))
    # step_range = 4

    # clear_directory("datas/fig")
    # for feature_pair in feature_pairs:
    #     for i in range(3, step_range):
    #         print(f"trainning...{feature_pair}")
    #         train_goodFeatures, train_goodLabels = get_train_features(goodFeatures, goodLabels, goodNames, feature_pair, i)
    #         train_badFeatures, train_badLabels = get_train_features(badFeatures, badLables, badNames, feature_pair, i)
    #         DumpSampleJson(train_goodFeatures, f"good_features.sudden_turn.{goodBehaviorTaskId}.{badBehaviorTaskId}.json", overwrite=True)
    #         DumpSampleJson(train_badFeatures, f"bad_features.sudden_turn.{goodBehaviorTaskId}.{badBehaviorTaskId}.json", overwrite=True)

    #         xgboost = XGBoost()
    #         accuracy = xgboost.Train(train_goodFeatures, train_badFeatures, train_goodLabels, train_badLabels, feature_pair + (f"step-{i}",))
    #         print(f"Accuracy: {accuracy}")
    # xgboost.Save(f"comfortobm_{feature_pair}_no_diff_step3.model")

    # Verify("sudden_turn.model")