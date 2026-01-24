# AUTO-GENERATED RULES BY NEUROCURSOR RULE MINER
# DO NOT EDIT MANUALLY UNLESS YOU KNOW WHAT YOU ARE DOING

def classify_gesture(features):
    # Unpack features for speed (and readability)
    thumb_bend = features['thumb_bend']
    index_bend = features['index_bend']
    mid_bend = features['mid_bend']
    ring_bend = features['ring_bend']
    pinky_bend = features['pinky_bend']
    pinch_dist = features['pinch_dist']
    thumb_spread = features['thumb_spread']
    mid_palm_dist = features['mid_palm_dist']
    orientation_y = features['orientation_y']
    orientation_x = features['orientation_x']

    if thumb_spread <= 0.9417:
        if index_bend <= 146.6366:
            if ring_bend <= 169.2113:
                if pinky_bend <= 4.2021:
                    if thumb_bend <= 36.6285:
                        if thumb_spread <= 0.3479:
                            return 'THE_DROP-1'
                        else:  # thumb_spread > 0.3479
                            return 'NOISE_RANDOM_5'
                    else:  # thumb_bend > 36.6285
                        return 'DELETE19_OPEN'
                else:  # pinky_bend > 4.2021
                    if orientation_y <= -0.1996:
                        if thumb_spread <= 0.5803:
                            if index_bend <= 128.2473:
                                if thumb_bend <= 41.1149:
                                    if mid_bend <= 55.7001:
                                        if pinky_bend <= 93.2582:
                                            return 'THE_DROP-2'
                                        else:  # pinky_bend > 93.2582
                                            return 'THE_BIRD_M_I_T'
                                    else:  # mid_bend > 55.7001
                                        if pinky_bend <= 27.8861:
                                            return 'SPIDER_MAN_2'
                                        else:  # pinky_bend > 27.8861
                                            return 'THE_POINT_1'
                                else:  # thumb_bend > 41.1149
                                    if orientation_x <= 0.0267:
                                        if orientation_x <= -0.1169:
                                            return 'V_GLUED_L'
                                        else:  # orientation_x > -0.1169
                                            return 'V_GLUED_M'
                                    else:  # orientation_x > 0.0267
                                        return 'V_GLUED_R'
                            else:  # index_bend > 128.2473
                                if orientation_y <= -0.2729:
                                    if mid_palm_dist <= 0.9645:
                                        if ring_bend <= 154.0514:
                                            return 'THE_GRAB_2'
                                        else:  # ring_bend > 154.0514
                                            return 'THE_GRAB_4'
                                    else:  # mid_palm_dist > 0.9645
                                        if thumb_spread <= 0.5012:
                                            return 'HITCHHIKER_RIGHT_CLOSED'
                                        else:  # thumb_spread > 0.5012
                                            return 'HITCHHIKER_RIGHT_CLOSED'
                                else:  # orientation_y > -0.2729
                                    if pinky_bend <= 76.3824:
                                        return 'THE_SCISSORS_1'
                                    else:  # pinky_bend > 76.3824
                                        return 'THE_PINCH_4'
                        else:  # thumb_spread > 0.5803
                            if mid_palm_dist <= 1.3019:
                                if ring_bend <= 60.9851:
                                    return 'DELETE19_CLOSED'
                                else:  # ring_bend > 60.9851
                                    if index_bend <= 134.6719:
                                        if ring_bend <= 116.4175:
                                            return 'THE_PINCH_5'
                                        else:  # ring_bend > 116.4175
                                            return 'THE_PINCH_4'
                                    else:  # index_bend > 134.6719
                                        if orientation_y <= -0.2829:
                                            return 'HITCHHIKER_RIGHT_OPEN'
                                        else:  # orientation_y > -0.2829
                                            return 'HITCHHIKER_RIGHT_OPEN'
                            else:  # mid_palm_dist > 1.3019
                                if index_bend <= 29.9896:
                                    if thumb_bend <= 13.8549:
                                        return 'CTRL_PINKY'
                                    else:  # thumb_bend > 13.8549
                                        return 'CTRL_PINKY'
                                else:  # index_bend > 29.9896
                                    if ring_bend <= 151.8711:
                                        return 'THE_BIRD_M_T'
                                    else:  # ring_bend > 151.8711
                                        return 'THE_BIRD_M_T'
                    else:  # orientation_y > -0.1996
                        if index_bend <= 2.7406:
                            if mid_bend <= 66.5895:
                                return 'NOISE_RANDOM_5'
                            else:  # mid_bend > 66.5895
                                return 'THE_SHHH'
                        else:  # index_bend > 2.7406
                            if pinky_bend <= 149.6238:
                                if mid_bend <= 71.1831:
                                    if orientation_y <= -0.1261:
                                        if orientation_x <= 0.0336:
                                            return 'NOISE_RANDOM_1'
                                        else:  # orientation_x > 0.0336
                                            return 'NOISE_RANDOM_4'
                                    else:  # orientation_y > -0.1261
                                        if orientation_x <= 0.0661:
                                            return 'NOISE_RANDOM_5'
                                        else:  # orientation_x > 0.0661
                                            return 'NOISE_RANDOM_4'
                                else:  # mid_bend > 71.1831
                                    if thumb_bend <= 24.6834:
                                        if orientation_y <= -0.1314:
                                            return 'SHAKA_2'
                                        else:  # orientation_y > -0.1314
                                            return 'SHAKA_1'
                                    else:  # thumb_bend > 24.6834
                                        if pinky_bend <= 77.7631:
                                            return 'NOISE_RANDOM_4'
                                        else:  # pinky_bend > 77.7631
                                            return 'NOISE_RANDOM_4'
                            else:  # pinky_bend > 149.6238
                                if thumb_bend <= 14.5917:
                                    if orientation_x <= -0.0933:
                                        return 'HITCHHIKER_LEFT_OPEN'
                                    else:  # orientation_x > -0.0933
                                        return 'THE_LINK_READY'
                                else:  # thumb_bend > 14.5917
                                    return 'HITCHHIKER_LEFT_CLOSED'
            else:  # ring_bend > 169.2113
                if orientation_x <= -0.0654:
                    return 'W_GLUED'
                else:  # orientation_x > -0.0654
                    if index_bend <= 3.5280:
                        return 'SPIDER_MAN_1'
                    else:  # index_bend > 3.5280
                        if index_bend <= 10.3055:
                            return 'THE_POINT_3'
                        else:  # index_bend > 10.3055
                            return 'NOISE_RANDOM_1'
        else:  # index_bend > 146.6366
            if thumb_spread <= 0.7523:
                if mid_palm_dist <= 0.7534:
                    return 'THE_SCISSORS_1'
                else:  # mid_palm_dist > 0.7534
                    return 'NOISE_RANDOM_2'
            else:  # thumb_spread > 0.7523
                return 'THE_LINK_READY'
    else:  # thumb_spread > 0.9417
        if orientation_y <= -0.1044:
            if orientation_x <= 0.1573:
                return 'GEN_Z_HEART'
            else:  # orientation_x > 0.1573
                return 'GEN_Z_HEART'
        else:  # orientation_y > -0.1044
            if orientation_y <= -0.0711:
                if ring_bend <= 141.5992:
                    return 'THE_LINK_CLUNCH'
                else:  # ring_bend > 141.5992
                    return 'THE_LINK_CLUNCH'
            else:  # orientation_y > -0.0711
                if ring_bend <= 92.5674:
                    return 'NOISE_RANDOM_5'
                else:  # ring_bend > 92.5674
                    return 'THE_LINK_READY'
    return 'UNKNOWN'