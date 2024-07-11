import openai
data = {
    "model": "lmsys/vicuna-7b-v1.5", 
    "prompt": [[
        1, 438, 3901, 294, 6692, 2261, 1192, 29871, 29896, 29929, 29899, 29906, 29947, 29889, 438, 3901, 294, 2669, 22306, 263, 29889, 13361, 1891, 19531, 414, 29889, 19019, 4285, 526, 4148, 1891, 304, 19531, 697, 975, 344, 294, 2669, 22306, 363, 1269, 29871, 29953, 489, 10874, 3785, 310, 6136, 14879, 2669, 408, 263, 4509, 310, 263, 501, 29889, 29903, 29889, 6692, 408, 18694, 2400, 29889, 29498, 29879, 310, 3109, 1135, 29871, 29953, 7378, 14385, 29892, 607, 6467, 28103, 278, 11780, 363, 278, 9862, 310, 975, 344, 294, 2669, 22306, 29892, 1122, 367, 12420, 491, 4417, 278, 1353, 310, 7378, 304, 8161, 6625, 8270, 2669, 11183, 278, 3001, 1353, 310, 975, 344, 294, 2669, 22306, 4148, 1891, 29889, 2391, 287, 6763, 10116, 322, 17140, 10116, 526, 20978, 573, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 25373, 4038, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29897, 4451, 2975, 8707, 3308, 29892, 1546, 29871, 29955, 5846, 29871, 29896, 29929, 29946, 29896, 322, 29871, 29906, 3839, 29871, 29896, 29929, 29946, 29953, 29889, 512, 20602, 975, 344, 294, 2669, 29892, 838, 16191, 338, 5545, 5377, 8707, 3308, 29889, 530, 975, 344, 294, 2669, 2594, 338, 451, 4148, 1891, 363, 263, 15958, 310, 263, 29871, 29953, 489, 10874, 3785, 29889, 313, 29906, 29897, 19109, 29892, 1546, 29871, 29906, 29955, 5306, 29871, 29896, 29929, 29945, 29900, 322, 29871, 29906, 29955, 5468, 29871, 29896, 29929, 29945, 29946, 29889, 24596, 277, 11183, 385, 975, 344, 294, 2669, 2594, 338, 4148, 1891, 363, 1269, 4098, 310, 6136, 14879, 2669, 408, 263, 4509, 310, 278, 501, 29889, 29903, 29889, 8811, 16330, 297, 278, 25373, 3495, 488, 3974, 4038, 297, 19109, 1546, 29871, 29896, 3786, 29871, 29896, 29929, 29953, 29947, 322, 29871, 29941, 29896, 3111, 29871, 29896, 29929, 29955, 29941, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 3495, 488, 3974, 5146, 4038, 526, 29115, 408, 3353, 7378, 29889, 960, 263, 19019, 631, 20586, 263, 4098, 310, 3495, 488, 3974, 5146, 363, 263, 3785, 29898, 29879, 29897, 310, 2669, 297, 19109, 29892, 769, 278, 19019, 631, 1122, 884, 7150, 16200, 363, 263, 6590, 4098, 7113, 9862, 310, 385, 975, 344, 294, 2669, 2594, 29889, 313, 29941, 29897, 18444, 29892, 1546, 29871, 29896, 5468, 29871, 29896, 29929, 29945, 29947, 322, 29871, 29906, 29947, 4779, 29871, 29896, 29929, 29955, 29941, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 18444, 526, 29115, 408, 3353, 7378, 363, 16200, 11183, 278, 975, 344, 294, 2669, 2594, 29889, 960, 263, 19019, 631, 20586, 263, 4098,310, 3495, 488, 3974, 5146, 363, 263, 3785, 29898, 29879, 29897, 310, 323, 29928, 29979, 2669, 297, 18444, 29892, 769,278, 19019, 631, 1122, 884, 7150, 16200, 363, 263, 6590, 4098, 7113, 9862, 310, 385, 975, 344, 294, 2669, 2594, 29889,313, 29946, 29897, 450, 13298, 2185, 8063, 29892, 1546, 29871, 29906, 29929, 3786, 29871, 29896, 29929, 29953, 29945, 322, 29871, 29906, 29896, 3839, 29871, 29896, 29929, 29953, 29953, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 13298, 2185, 8063, 526, 29115, 408, 3353, 7378, 29889, 313, 29945, 29897, 997, 359, 29892, 1546, 29871, 29896, 5490, 29871, 29896, 29929, 29953, 29953, 322, 29871, 29906, 29947, 4779, 29871, 29896, 29929, 29955, 29941, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 997, 359, 526, 29115, 408, 3353, 7378, 29889, 313, 29953, 29897, 9287, 26942, 1546, 29871, 29896, 5490, 29871, 29896, 29929, 29955, 29896, 322, 29871, 29906, 29947, 4779, 29871, 29896, 29929, 29955, 29941, 29889, 5196, 9139, 1818, 4021, 1598, 363, 3495, 488, 3974, 5146, 304, 7150, 16200, 363, 385, 975, 344, 294, 2669, 2594, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 3495, 488, 3974, 5146, 4038, 526, 29115, 408, 3353, 7378, 29889, 313, 29955, 29897, 9388, 19930, 29892, 1546, 29871, 29953, 3111, 29871, 29896, 29929, 29947, 29941, 322, 29871, 29906, 29946, 3786, 29871, 29896, 29929, 29947, 29946, 29892, 363, 278, 1023, 10340, 9904, 297,14880, 29871, 29896, 29929, 489, 29896, 29955, 29890, 29898, 29953, 467, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 9388, 19930, 526, 29115, 408, 3353, 7378, 29889, 313, 29947, 29897, 450, 9034, 713, 402, 16302, 1546, 29871, 29906, 29955, 5468, 29871, 29896, 29929, 29947, 29955, 322, 29871, 29896, 3111, 29871, 29896, 29929, 29929, 29900, 29892, 363, 20462, 382, 2753, 342, 2811, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 9034, 713, 402, 16302, 526, 29115, 408, 3353, 7378, 29889, 313, 29929, 29897, 450, 9034, 713, 402, 16302, 1546, 29871, 29896, 29955, 5490, 29871, 29896, 29929, 29929, 29896, 322, 29871, 29941, 29896, 3111, 29871, 29896, 29929, 29929, 29941, 29892, 363, 20462,2726, 814, 24444, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 9034, 713, 402, 16302, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29900, 29897, 1260, 22194, 29892, 1546, 29871, 29896, 5490, 29871, 29896, 29929, 29947, 29896, 322, 29871, 29896, 6339, 29871, 29896, 29929, 29929, 29906, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 1260, 22194, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29896, 29897, 6254, 19627, 29892, 1546, 29871, 29945, 5846, 29871, 29896, 29929, 29929, 29906, 322, 29871, 29941, 29896, 4779, 29871, 29896, 29929, 29929, 29945, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 6254, 19627, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29906, 29897, 3455, 12654, 362, 297, 438, 29638, 29892, 297, 278, 315, 3919, 19795, 4038, 310, 6931, 29892, 322, 1090, 278, 2761, 310, 278, 422, 10222, 424, 27317, 29892, 315, 3919, 19795, 29892, 1546, 29871, 29896, 29896, 3839, 29871, 29906, 29900, 29900, 29896, 322, 29871, 29941, 29896, 5846, 29871, 29906, 29900, 29896, 29946, 29936, 438, 29638, 29899, 4819, 2638, 407, 1475, 29892, 297, 278, 26260, 29892, 1546, 29871, 29896, 29929, 3839, 29871, 29906, 29900, 29900, 29896, 322, 29871, 29941, 29896, 5846, 29871, 29906, 29900, 29896, 29946, 29936, 438, 29638, 29899, 29950, 1398, 310, 10557, 29892, 297, 27467, 747, 449, 29875, 29892, 1546, 29871, 29896, 5490, 29871, 29906, 29900, 29900, 29947, 322, 29871, 29941, 29896, 5846, 29871, 29906, 29900, 29896, 29946, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 26260, 29892, 27467, 747, 449, 29875, 29892, 470, 278, 315, 3919, 19795, 4038, 310, 6931, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29941, 29897, 3455, 12654, 362, 297, 438, 6545, 29892, 297, 278, 315, 3919, 19795, 4038, 310, 6931, 29892, 322, 1090, 278, 2761, 310, 278, 422, 10222, 424, 27317, 29892, 315, 3919, 19795, 29892, 1546, 29871, 29896, 29929, 4779, 29871, 29906, 29900, 29900, 29941, 322, 29871, 29941, 29896, 3111, 29871, 29906, 29900, 29896, 29900, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 315, 3919, 19795, 4038, 310, 6931, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29946, 29897, 3455, 12654, 362, 297, 438, 2797, 297, 278, 315, 3919, 19795, 4038, 310, 6931, 29892, 322, 1090, 278, 2761, 310, 278, 422, 10222, 424, 27317, 29892, 315, 3919, 19795, 29892, 1546, 29871, 29896, 3839, 29871, 29906, 29900, 29896, 29900, 322, 29871, 29941, 29896, 5846, 29871, 29906, 29900, 29896, 29896, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515, 278, 315, 3919, 19795, 4038, 310, 6931, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29945, 29897, 3455, 12654, 362, 297, 438, 8193, 29892, 297, 278, 315, 3919, 19795, 4038, 310, 6931, 29892, 322, 1090, 278, 2761, 310, 278, 422, 10222, 424, 27317, 29892, 315, 3919, 19795, 29892, 1546, 29871, 29896, 29945, 5306, 29871, 29906, 29900, 29896, 29946, 322, 263, 2635, 304, 367, 10087, 29889, 450, 7378, 310, 18517, 304, 29892, 322, 25619, 515,278, 315, 3919, 19795, 4038, 310, 6931, 526, 29115, 408, 3353, 7378, 29889, 313, 29896, 29953, 29897, 3455, 12654, 362, 297, 8079, 29903, 29892, 297, 278, 315, 3919, 19795, 4038, 310, 6931, 29892, 322, 1090, 278, 2761, 310, 278, 422, 10222, 424, 27317, 29892, 315, 3919, 19795, 29892, 470, 27467, 747, 449, 29875, 29892, 319, 15860, 2965, 6488, 29892, 1546, 29871, 29896, 5490, 29871, 29906, 29900, 29896, 29945, 322, 263, 2635, 304, 367, 10087, 29889, 450, 7378, 310, 18517,304, 29892, 322, 25619, 515, 27467, 747, 449, 29875, 470, 278, 315, 3919, 19795, 4038, 310, 6931, 526, 29115, 408, 3353, 7378, 29889, 289, 29889, 1128, 28043, 29889, 2823, 21330, 25281, 29871, 29953, 29955, 29900, 489, 29896, 29889, 13, 16492, 29901, 437, 366, 679, 975, 344, 294, 2669, 22306, 363, 413, 487, 29874, 29973, 13, 22550, 29901, 694]], 
    "echo": True, 
    "logprobs": 10, 
    "max_tokens": 0, 
    "seed": 1234, 
    "temperature": 0.0
}

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="sk-1234"
)
res = client.completions.create(**data)
print(res)