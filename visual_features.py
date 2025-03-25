from collections.abc import Iterable
from typing import Callable
import inspect
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
import cv2
import numpy as np

global_lower = np.array([0, 0, 0])  # Нижняя граница первого диапазона
global_upper = np.array([255, 255, 255])  # Верхняя граница первого диапазона

def make_kwargs(is_kwargs: bool, param_names: list[str], args: list, wargs: list, params: dict, wparams: dict):
    kwargs = {}
    
    if is_kwargs:
        kwargs = {
            **wparams,
            **params
        }
    
    args_i = 0
    wargs_i = 0
    for param_name in param_names:
        if param_name in params:
            kwargs[param_name] = params[param_name]
        elif param_name in wparams:
            kwargs[param_name] = wparams[param_name]
        elif args_i < len(args):
            kwargs[param_name] = args[args_i]
            args_i += 1
        elif wargs_i < len(wargs):
            kwargs[param_name] = wargs[wargs_i]
            wargs_i += 1
        
    # print("param_names:", param_names)
    # print("args:", len(args))
    # print("wargs:", len(wargs))
    # print("params:", params.keys())
    # print("wparams:", wparams.keys())
    return kwargs
            

    # param_names: ['key', 'kwargs']
    # args: ()
    # wargs: ()
    # params: dict_keys(['image', 'gray', 'hsv_image', 'R', 'G', 'B', 'H', 'S', 'V', 'image_height', 'image_width', 'update_params'])
    # wparams: dict_keys(['key'])
    
    # assert (len(args) + len(params)) <= len(param_names)
    assert (len(wargs) + len(wparams)) <= len(param_names)
    kwargs = {
        **wparams,
        **params
    }
    kwargs = {key:value for key, value in kwargs.items() if key in param_names}

    # for param_name, warg in zip(param_names, wargs):
    #     assert not (param_name in wparams)
    # for param_name, arg in zip(param_names, args):
    #     assert not (param_name in params)

    args = args + wargs
    for param_name, arg in zip(param_names, args):
        print("########################")
        print(kwargs)
        assert not (param_name in kwargs), param_name
        kwargs[param_name] = arg

    assert len(kwargs) <= len(param_names)
    
    return kwargs


def pipeline_element(main_func: Callable) -> Callable:
    sig = inspect.signature(main_func)
    parameters = sig.parameters
    kwargs_names = [
        name for name in sig.parameters if sig.parameters[name].kind == 4
    ]
    param_names = list(parameters.keys())
    def pipeline_element_wrapper(*wargs, **wparams):
        def func(*args, **params):
            kwargs = make_kwargs(
                len(kwargs_names) > 0,
                param_names, 
                args, 
                wargs, 
                params, 
                wparams
            )
            return main_func(**kwargs)

        wparams_str = []
        for key, value in wparams.items():
            wparams_str.append(f"{key}={value}")
        wparams_str = ",".join(wparams_str)
        func.__name__ = f"{main_func.__name__}({wparams_str})"
        return func
    return pipeline_element_wrapper


########################
@pipeline_element
def denoise(image, d=9, sigma_color=75, sigma_space=75):
    bilateralFilter = cv2.bilateralFilter(
        image,
        d=d, 
        sigmaColor=sigma_color, 
        sigmaSpace=sigma_space
    )
    return bilateralFilter

@pipeline_element
def text_recognition(image, padding=0, padding_value=0):
    height, width = image.shape[:2]
    is_2d = len(image.shape) == 2

    if padding:
        if is_2d:
            canvas = np.full((height + 2 * padding, width + 2 * padding), padding_value, dtype=image.dtype)
        else:
            canvas = np.full((height + 2 * padding, width + 2 * padding, 3), padding_value, dtype=image.dtype)
        canvas[padding:-padding, padding:-padding] = image
        image = canvas

    paddle_ocr_results = paddle_ocr.ocr(image)[0]
    paddle_ocr_results = [] if paddle_ocr_results is None else paddle_ocr_results
    ocr_results = []
    for res in paddle_ocr_results:
        x1 = min(res[0][i][0] for i in range(4)) - padding
        x2 = max(res[0][i][0] for i in range(4)) - padding
        y1 = min(res[0][i][1] for i in range(4)) - padding
        y2 = max(res[0][i][1] for i in range(4)) - padding

        if digit.isdigit():
            ocr_results.append({
                "text": text,
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })
    return ocr_results

@pipeline_element
def color_MeanShift_clustering(image, quantile=0.2, n_samples=500):
    pixels = image.reshape((-1, 3))

    # Оценка оптимальной ширины ядра (bandwidth)
    bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=n_samples)
    
    # Применение Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels)
    
    # Получение меток кластеров и центров
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    # Замена пикселей на цвета центров кластеров
    segmented_image = cluster_centers[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image

@pipeline_element
def segment_image(image):
    height, width, _ = image.shape

    # 2. Преобразуем изображение в 5D-пространство (цвет + координаты)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    feature_space = np.column_stack((image.reshape(-1, 3), X.ravel(), Y.ravel()))

    # 3. Оценка оптимального радиуса (bandwidth) и применение Mean Shift
    bandwidth = estimate_bandwidth(feature_space, quantile=0.1, n_samples=500)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(feature_space)
    
    # Получаем метки кластеров
    labels = mean_shift.labels_
    num_clusters = len(np.unique(labels))

    # 4. Создание сегментированного изображения
    segmented_image = np.zeros_like(image.reshape(-1, 3))
    cluster_colors = np.random.randint(0, 255, (num_clusters, 3))  # Цвета кластеров
    for i, label in enumerate(labels):
        segmented_image[i] = cluster_colors[label]
    
    segmented_image = segmented_image.reshape(height, width, 3)

    # 5. Генерация масок для каждого кластера
    masks = []
    for cluster_id in range(num_clusters):
        mask = (labels.reshape(height, width) == cluster_id).astype(np.uint8) * 255
        masks.append(mask)

    return (segmented_image, masks)

@pipeline_element
def get_threshold(map_, threshold_value=127, inverse=False):
    # print(map_.shape)
    if inverse:
        flag = cv2.THRESH_BINARY_INV
    else:
        flag = cv2.THRESH_BINARY
    
    ret, thresh = cv2.threshold(map_, threshold_value, 255, flag)
    return thresh

@pipeline_element
def get_adaptive_threshold(map_, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inverse=False, blockSize=11, C=2):
    if inverse:
        flag = cv2.THRESH_BINARY_INV
    else:
        flag = cv2.THRESH_BINARY
    
    mask = cv.adaptiveThreshold(
        map_,
        255,
        method,
        flag,
        blockSize,
        C
    )
    return mask

@pipeline_element
def dilate(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask

@pipeline_element
def erode(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    mask = cv2.erode(mask, kernel, iterations=iterations)
    return mask

@pipeline_element
def opening(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=iterations
    )
    return mask

@pipeline_element
def closing(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(
        mask, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=iterations
    )
    return mask

@pipeline_element
def negate_mask(mask):
    return ~mask

@pipeline_element
def unite_masks(masks):
    result = np.zeros_like(masks[0])
    for mask in masks:
        result = cv2.bitwise_or(result, mask)
    return result

@pipeline_element
def intersect_masks(masks):
    result = np.ones_like(masks[0])
    for mask in masks:
        result = cv2.bitwise_and(result, mask)
    return result

@pipeline_element
def subtract_masks(masks):
    mask1, mask2 = masks
    result = cv2.bitwise_and(mask1, ~mask2)
    return result


@pipeline_element
def get_sobel(gray, ksize=3):
    gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    sobel = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    return sobel

@pipeline_element
def get_edges(gray, blur_value=None, aperture_size=5, t_lower=10, t_upper=200):
    if blur_value:
        gray = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

    edges = cv2.Canny(
        gray, t_lower, t_upper, apertureSize=aperture_size
    )
    return edges

@pipeline_element
def get_hsv_color_mask(hsv_image, hue_lower, hue_upper, saturation_lower=0, saturation_upper=255, value_lower=0, value_upper=255):
    lower = np.array([hue_lower, saturation_lower, value_lower])
    upper = np.array([hue_upper, saturation_upper, value_upper])
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask

@pipeline_element
def get_rgb_color_mask(image, lower_values, upper_values):
    lower = np.array(lower_values)
    upper = np.array(upper_values)
    mask = cv2.inRange(image, lower, upper)
    return mask

@pipeline_element
def get_color_map(image, color):
    max_diff = (
        max(color[0], 255 - color[0]),
        max(color[1], 255 - color[1]),
        max(color[2], 255 - color[2]),
    )
    color = np.array(color).reshape(1, 1, 3)
    max_diff = np.array(max_diff).reshape(1, 1, 3)
    
    diff = image.astype(np.int32) - color
    diff = np.abs(diff)
    
    diff = diff / max_diff
    diff = diff.mean(axis=-1)
    diff = diff * 255
    
    diff = diff.astype(np.uint8)
    color_map = 255 - diff

    return color_map

@pipeline_element
def get_colorful_mask(S, lower_value=100):
    mask = cv2.inRange(S, lower_value, 255)
    return mask

@pipeline_element
def get_V_white_mask(V, lower_value=220):
    mask = cv2.inRange(V, lower_value, 255)
    return mask

@pipeline_element
def get_V_black_mask(V, upper_value=40):
    mask = cv2.inRange(V, 0, upper_value)
    return mask

@pipeline_element
def get_rgb_white_mask(image, lower_value=200):
    global global_upper
    lower = np.array([lower_value, lower_value, lower_value])
    mask = cv2.inRange(image, lower, global_upper)
    return mask

@pipeline_element
def get_rgb_black_mask(image, upper_value=40):
    global global_lower
    upper = np.array([upper_value, upper_value, upper_value])
    mask = cv2.inRange(image, global_lower, upper)
    return mask

@pipeline_element
def get_gray_gradient_mask(R, G, B, tolerance=15):
    R = R.astype(np.int16)
    G = G.astype(np.int16)
    B = B.astype(np.int16)
    
    mask = (np.abs(R - G) <=  tolerance) & (np.abs(R - B) <=  tolerance) & (np.abs(B - G) <=  tolerance)
    mask = mask.astype(np.uint8) * 255
    return mask

@pipeline_element
def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

@pipeline_element
def filter_contours(contours, image_height, image_width, area_norm_limits=None, polygon_limits=None):
    image_area = image_height * image_width

    filtered_contours = []
    for cnt in contours:
        if area_norm_limits:
            area_norm = cv2.contourArea(cnt) / image_area

            if area_norm < area_norm_limits[0] or area_norm_limits[1] < area_norm:
                continue

        if polygon_limits:
            approx = cv2.approxPolyDP( 
                contour, 0.01 * cv2.arcLength(contour, True), True
            )
            n = len(approx)

            if n < polygon_limits[0] or polygon_limits[1] < n:
                continue

        filtered_contours.append(cnt)

    return filtered_contours

@pipeline_element
def grabcut(image, rect, iterCount=10, mode=cv2.GC_INIT_WITH_RECT):
    # Матрица меток размером, совпадающим с img.shape[:2] (высота × ширина).
    # Должна быть типа np.uint8 и содержать значения:
    # cv2.GC_BGD (0) – явно задний план.
    # cv2.GC_FGD (1) – явно передний план.
    # cv2.GC_PR_BGD (2) – вероятный задний план.
    # cv2.GC_PR_FGD (3) – вероятный передний план.
    # Изначально задаётся пользователем (например, прямоугольником), затем обновляется в процессе выполнения алгоритма.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Координаты прямоугольника (x, y, w, h), внутри которого предполагается передний план.
    # Используется только при mode=cv2.GC_INIT_WITH_RECT.
    # Пиксели внутри прямоугольника помечаются как вероятный передний план, а снаружи — как явный задний план.

    # Большее число итераций может улучшить качество сегментации.

    # Определяет способ инициализации:
    # cv2.GC_INIT_WITH_RECT – инициализация с помощью прямоугольника rect.
    # cv2.GC_INIT_WITH_MASK – инициализация с помощью маски mask (полезно для доработки результатов).
    # cv2.GC_EVAL – продолжение работы алгоритма на основе уже полученной маски.
    # mode – (int, необязательный, по умолчанию cv2.GC_INIT_WITH_RECT)

    # Временный массив (размера (1, 65), np.float64), 
    # который используется для хранения модели фона (гауссовых распределений).
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    
    (mask, bgModel, fgModel) = cv2.grabCut(
        image, 
        mask, 
        rect, 
        bgModel,
    	fgModel, 
        iterCount=iterCount, 
        mode=mode
    )

    # Обновлённая маска сегментации (np.uint8).
    # Может содержать значения:
    # 0 (cv2.GC_BGD) – фон.
    # 1 (cv2.GC_FGD) – передний план.
    # 2 (cv2.GC_PR_BGD) – вероятный фон.
    # 3 (cv2.GC_PR_FGD) – вероятный передний план.
    # Для получения итоговой бинарной маски можно использовать:
    # binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # bgdModel – (numpy.ndarray)
    # Обновлённая модель фона (используется внутри алгоритма).
    # fgdModel – (numpy.ndarray)
    # Обновлённая модель переднего плана.
    return mask

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
# cv2.FLOODFILL_FIXED_RANGE
floodflags |= (255 << 8)

@pipeline_element
def floodfill(
    image, 
    point,
    flags= 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8), # cv2.FLOODFILL_FIXED_RANGE
    loDiff=(10, 10, 10),
    upDiff=(10, 10, 10),
):
    h, w = image.shape[:2]
    mask = np.zeros((h + 2,w + 2), np.uint8)
    
    retval, image, _, rect = cv2.floodFill(
        image, 
        mask, 
        point, 
        255,
        flags=flags,
        loDiff=loDiff,
        upDiff=upDiff,
    )
    mask = mask[0:-2, 1:-1]
    return mask
    
#### Funcs ####
def get_dist(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

def closeness_func(target_point, **kwargs):
    def calc_dist(cnt):
        M = cv2.moments(cnt)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        return get_dist(center, target_point)
    return calc_dist
    
def max_area_func(*args, **kwargs):
    return cv2.contourArea

criteria = {
    "max_area": max_area_func,
    "closeness": closeness_func,
}
#################
@pipeline_element
def get_best_contour(contours, criterion="max_area", **kwargs):
    if isinstance(criterion, str):
        criterion = criteria[criterion](**kwargs)
    best_contour = max(contours, key=criterion)
    return best_contour
    
@pipeline_element
def detect_circles(gray, blurness=None, dp=1, minDist=20, param1=50, param2=30, minRadius=15, maxRadius=60):
    if blurness:
        blurred = cv2.medianBlur(gray, blurness)
    else:
        blurred = gray

    # Используем метод Хафа для поиска окружностей
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,              # Параметр "dp" (разрешение аккумулятора)
        minDist=minDist,          # Минимальное расстояние между центрами окружностей
        param1=param1,           # Верхний порог для Canny
        param2=param2,           # Порог для метода Хафа (чем меньше, тем больше ложных окружностей)
        minRadius=minRadius,         # Минимальный радиус окружности
        maxRadius=maxRadius,         # Максимальный радиус окружности
    )

    # Если окружности найдены, извлекаем их центры
    if circles is None:
        return []
    
    circles = np.round(circles[0, :]).astype("int")  # Округляем до целых чисел
    # centers = [(x, y) for x, y, r in circles]       # Извлекаем координаты центров
    return circles

@pipeline_element
def detect_lines(edges, rho=1, threshold=40, minLineLength=None, maxLineGap=None):
    cv2_lines = cv2.HoughLinesP(edges, rho, np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if cv2_lines is None:
        return []
    lines = cv2_lines[:, 0].tolist()
    return lines

@pipeline_element
def detect_corners(
    gray, 
    threshold1=120, 
    threshold2=255, 
    apertureSize=3, 
    L2gradient=False, 
    maxCorners=27, 
    qualityLevel=0.01,
    minDistance=10,
    mask=None,
    blockSize=3,
    useHarrisDetector=False,
):
    # Границы, у которых градиент ниже этого значения, отбрасываются.
    # Границы, у которых градиент выше этого значения, точно считаются границами.
    # Границы, у которых градиент между threshold1 и threshold2, включаются в результат, только если связаны с пикселями, превышающими верхний порог.
    # размер ядра оператора Собеля
    # True – используется более точное Евклидово расстояние

    # Максимальное количество углов, которые нужно найти.
    # Минимальный допустимый уровень качества угла относительно лучшего найденного угла.
    # Например, если qualityLevel=0.01, то точки с откликом меньше 0.01 * max_response отбрасываются.
    # Только пиксели, соответствующие ненулевым значениям маски, будут участвовать в поиске углов.
    # Чем больше значение, тем более сглаженными будут градиенты. (Sobel)
    # Если True, используется детектор Харриса вместо Shi-Tomasi

    canny = cv2.Canny(gray, threshold1, threshold2, apertureSize) # L2gradient

    cv2_corners = cv2.goodFeaturesToTrack(
        canny, 
        maxCorners=maxCorners, 
        qualityLevel=qualityLevel, 
        minDistance=minDistance,
        mask=mask,
        blockSize=blockSize,
        useHarrisDetector=useHarrisDetector,
    )
    corners = []
    for corner in cv2_corners:
        x, y = corner.ravel()
        corners.append([int(x), int(y)])
    return corners

@pipeline_element
def visualize_contour(
    contour,
    image_height,
    image_width,
):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask

@pipeline_element
def visualizer(
    image,
    data,
    shape_type,
    color=(255, 0, 0),
    thickness=1,
):
    image = image.copy()
    if "rectangles" == shape_type:
        rectangles = data
        for (x1, y1, x2, y2) in rectangles:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
    elif "circles" == shape_type:
        circles = data
        for x, y, r in circles:
            cv2.circle(image, (x, y), r, color, thickness)

    elif "lines" == shape_type:
        lines = data
        for (x1, y1, x2, y2) in lines:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    elif "contours" == shape_type:
        contours = data
        cv2.drawContours(image, contours, -1, color, thickness)

    elif "best_contour" == shape_type:
        best_contour = data
        cv2.drawContours(image, [best_contour], -1, (255, 255, 255), thickness)

    elif "corners" == shape_type:
        corners = data
        for x, y in corners:
            cv2.circle(image, (x, y), thickness, color, -1)

    elif "ocr" == shape_type:
        ocr_results = data
        for ocr_result in ocr_results:
            x1, y1, x2, y2 = ocr_result["box"]
            text = ocr_result["text"]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness * 2, cv2.LINE_AA)

    return image

# @pipeline_element
def get_basic_features(image):
    R, G, B = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv_image = cv2.cvtColor(image,  cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)
    return {
        "image": image,
        "gray": gray,
        "hsv_image": hsv_image,
        "R": R,
        "G": G,
        "B": B,
        "H": H,
        "S": S,
        "V": V,
        "image_height": image.shape[0],
        "image_width": image.shape[1],
        "update_params": True
    }



def parallel(funcs_list: list[Callable]) -> Callable:
    def handler(essence):
        results = []
        for func in funcs_list:
            result = func(essence)
            results.append(result)
        return results
    return handler

def sequential(funcs_list: list[Callable]) -> Callable:
    def handler(essence):
        for func in funcs_list:
            essence = func(essence)
        return essence
    return func

@pipeline_element
def unite_dicts(dicts):
    result = {}
    for dict_ in dicts:
        result.update(dict_)
    return result

@pipeline_element
def get(key, **kwargs):
    return kwargs[key]

@pipeline_element
def dict_wrap(elements, keys):
    assert len(elements) == len(keys)
    return {key:value for key, value in zip(keys, elements)}

def execute(obj, args=[], params={}):
    # print(str(obj), args)
    params = params.copy()
    if isinstance(obj, tuple):
        for func in obj:
            args, params = execute(func, args, params)
        return args, params
    elif isinstance(obj, list):
        answers = []
        all_new_params = {}
        for func in obj:
            answer, new_params = execute(func, args, params)
            answers.append(answer[0])
            all_new_params.update(new_params)
        params.update(all_new_params)
        return [answers], params
            
    elif callable(obj):
        if obj.__name__ == "pipeline_element_wrapper":
            obj = obj()
        
        result = obj(*args, **params)
        if isinstance(result, dict):
            if result.get("update_params", False):
                params.update(result)
        return [result], params

    elif isinstance(obj, dict):
        params.udpate(obj)
        return args, params

    raise ValueError("Wrong object type (tuple, list, function)!")

#######
def get_features(image, funcs_with_params):
    params = get_basic_features(image)
    features = []
    for obj in funcs_with_params:
        feature, new_params = execute(obj, params=params)
        features.append(feature[0])
    
    return features
