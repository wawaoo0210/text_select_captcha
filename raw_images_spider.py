# coding=utf-8
import base64
import os
import random
import time
from pathlib import Path
from typing import List, Any

import cv2
import ddddocr
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 常量定义
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_CHANNEL_DIR = Path("channel")
RAW_IMAGE_DIR = Path("./images_raw")
BASE64_PREFIX = "data:image/png;base64,"
MIN_BOX_SIZE = 10
ASPECT_RATIO_LIMIT = 8
RANDOM_SLEEP_RANGE = (2, 5)


class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def save_v_channel(image_path: Path, output_dir: Path = DEFAULT_CHANNEL_DIR) -> Path:
        """保存图像的明度通道"""
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        output_path = output_dir / f"v_channel_{image_path.name}"
        cv2.imwrite(str(output_path), hsv[:, :, 2])
        print(f"已保存明度通道到：{output_path}")
        return output_path

    @staticmethod
    def process_image(image_path: Path, detector: ddddocr.DdddOcr, recognizer: ddddocr.DdddOcr) -> List[tuple[str, tuple]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        bboxes = detector.detection(image_bytes)
        # ✅ 按 x1 坐标从小到大排序
        bboxes.sort(key=lambda box: box[0])
        results = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            if any([w < MIN_BOX_SIZE, h < MIN_BOX_SIZE, max(w / h, h / w) > ASPECT_RATIO_LIMIT]):
                continue

            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            text = recognizer.classification(cv2.imencode('.png', cropped)[1].tobytes())
            if text.strip():
                results.append((text, bbox))  # 同时返回文本和坐标
                ImageProcessor._draw_result(img, x1, y1, x2, y2, text)

        result_path = DEFAULT_OUTPUT_DIR / f"result_{image_path.name}"
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(result_path), img)
        print(f"🎯 标注结果已保存至：{result_path}")
        return results

    @staticmethod
    def _draw_result(img: cv2.Mat, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


class WebCrawler:
    """网页爬取工具类"""

    def __init__(self):
        self.driver = self._init_driver()
        self.detector = ddddocr.DdddOcr(det=True)
        self.recognizer = ddddocr.DdddOcr()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @staticmethod
    def _init_driver() -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        try:
            service = Service(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=options)
        except Exception as e:
            print(f"❌ WebDriver 启动失败: {e}")
            raise

    def crawl_images(self, retry_limit: int = 3) -> tuple[list[Any], Any] | list[Any]:
        """保存网页上的图像并返回图像路径列表"""
        RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        try:
            self.driver.get("https://amap.pythonanywhere.com/word/")
            for _ in range(retry_limit):
                element = self._wait_for_elements()
                img_path, prompt = self._process_page(element)
                if img_path:
                    saved_paths.append(img_path)
                    return saved_paths, prompt  # 成功一次就返回
                self._random_delay()
        except Exception as e:
            print(f"❌ 爬取失败: {str(e)}")
        return saved_paths

    def _wait_for_elements(self) -> WebElement:
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//img'))
        )
        return WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, 'prompt'))
        )

    def _process_page(self, prompt_element: WebElement) -> tuple[Path, str] | None:
        img_element = self.driver.find_element(By.XPATH, '//img')
        WebDriverWait(self.driver, 10).until(
            lambda d: img_element.get_attribute('src') not in [None, '']
        )
        img_src = img_element.get_attribute('src')

        if not img_src.startswith(BASE64_PREFIX):
            print("❌ 未找到base64图片")
            return None

        prompt_text = prompt_element.text.replace(" ", "")
        if self._save_image(img_src, prompt_text):
            safe_name = "".join(c for c in prompt_text if c.isalnum() or c in ("_", "-"))
            return RAW_IMAGE_DIR / f"{safe_name}.png", prompt_text
        return None

    def _save_image(self, img_src: str, prompt_text: str) -> bool:
        base64_str = img_src.replace(BASE64_PREFIX, '')
        img_data = base64.b64decode(base64_str)

        safe_name = "".join(c for c in prompt_text if c.isalnum() or c in ("_", "-"))
        img_path = RAW_IMAGE_DIR / f"{safe_name}.png"

        try:
            with open(img_path, 'wb') as f:
                f.write(img_data)
            print(f"✅ 成功保存: {img_path}")
            return True
        except IOError as e:
            print(f"❌ 保存失败: {str(e)}")
            return False

    def _process_image(self, prompt_text: str) -> bool:
        img_path = RAW_IMAGE_DIR / f"{prompt_text}.png"
        try:
            channel_path = ImageProcessor.save_v_channel(img_path)
            results = ImageProcessor.process_image(channel_path, self.detector, self.recognizer)
            print("🧠 识别结果:", results)
            return True
        except Exception as e:
            print(f"❌ 图像处理失败: {str(e)}")
            return False

    def _random_delay(self) -> None:
        time.sleep(random.uniform(*RANDOM_SLEEP_RANGE))
        self.driver.refresh()

    def _simulate_clicks(self, image_path: Path, click_sequence: List[tuple]) -> None:
        """使用预存坐标进行点击，模拟更接近人类的点击行为"""
        img_element = self.driver.find_element(By.XPATH, '//img')
        img_rect = self.driver.execute_script(
            "return arguments[0].getBoundingClientRect();", img_element)

        img = cv2.imread(str(image_path))
        original_h, original_w = img.shape[:2]

        scale_x = img_rect['width'] / original_w
        scale_y = img_rect['height'] / original_h

        for text, bbox in click_sequence:
            if not bbox:
                print(f"❌ 缺失点击坐标: {text}")
                continue

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2 * scale_x
            center_y = (y1 + y2) / 2 * scale_y

            # 添加小幅随机偏移，模拟不精准点击
            offset_x = random.uniform(-7, 7)
            offset_y = random.uniform(-7, 7)
            center_x += offset_x
            center_y += offset_y

            abs_x = img_rect['left'] + center_x
            abs_y = img_rect['top'] + center_y

            # 模拟“人类”犹豫动作：先错一点，再回来
            if random.random() < 0.3:
                dx = random.uniform(-30, 30)
                dy = random.uniform(-30, 30)
                webdriver.ActionChains(self.driver).move_by_offset(dx, dy).perform()
                time.sleep(random.uniform(0.2, 0.4))

            # 模拟移动轨迹（分步移动）
            steps = random.randint(5, 10)
            for i in range(steps):
                inter_x = (center_x / steps) * (i + 1)
                inter_y = (center_y / steps) * (i + 1)
                webdriver.ActionChains(self.driver).move_to_element_with_offset(img_element, inter_x, inter_y).perform()
                time.sleep(random.uniform(0.01, 0.03))

            # 等一下，再点击
            time.sleep(random.uniform(0.3, 0.6))

            # 同时触发完整事件链（JS）
            self.driver.execute_script("""
                let el = document.elementFromPoint(arguments[0], arguments[1]);
                ['mousemove', 'mousedown', 'mouseup', 'click'].forEach(type => {
                    el.dispatchEvent(new MouseEvent(type, {
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: arguments[0],
                        clientY: arguments[1]
                    }));
                });
            """, abs_x, abs_y)

            # 点击后等一下再点下一次
            time.sleep(random.uniform(0.3, 0.5))
            print(f"✅ 模拟点击 {text} at ({center_x:.1f}, {center_y:.1f})")

    def refresh_page(self):
        """刷新页面"""
        self.driver.refresh()


def fill_click_sequence(results, prompt):
    click_sequence = []
    used_bboxes = set()

    for char in prompt:
        found = next(((t, b) for t, b in results if t == char and str(b) not in used_bboxes), None)
        if found:
            used_bboxes.add(str(found[1]))
        click_sequence.append(found or (char, None))

    print("📌 初步匹配:", click_sequence)

    # 补充自定义模型识别
    if any(b is None for _, b in click_sequence):
        alt_results = ImageProcessor.process_image(v_channel_path, detector, myocr)
        for i, (char, bbox) in enumerate(click_sequence):
            if bbox is None:
                found = next(((t, b) for t, b in alt_results if t == char and str(b) not in used_bboxes), None)
                if found:
                    click_sequence[i] = found
                    used_bboxes.add(str(found[1]))

    print("🔁 补充后:", click_sequence)

    # 随机填充空白项，使用未使用的 bboxes
    remaining_bboxes = [b for _, b in results if str(b) not in used_bboxes]
    random.shuffle(remaining_bboxes)

    for i, (char, bbox) in enumerate(click_sequence):
        if bbox is None and remaining_bboxes:
            chosen_bbox = remaining_bboxes.pop()
            click_sequence[i] = (char, chosen_bbox)
            used_bboxes.add(str(chosen_bbox))

    print("✅ 最终点击序列:", click_sequence)

    return click_sequence


if __name__ == "__main__":
    if not os.path.exists("images_raw"):
        os.makedirs("images_raw")
    if not os.path.exists("channel"):
        os.makedirs("channel")
    if not os.path.exists("output"):
        os.makedirs("output")

    detector = ddddocr.DdddOcr(det=True)
    recognizer = ddddocr.DdddOcr()

    myocr = ddddocr.DdddOcr(det=True,
                            import_onnx_path="models/click_captcha_0.65625_474_38000_2025-04-16-15-14-11.onnx",
                            charsets_path="models/charsets.json")

    for i in range(10):
        print(f"\n🔁 开始第 {i + 1} 次验证")

        is_recognized = False

        with WebCrawler() as crawler:


            while not is_recognized:
                result = crawler.crawl_images()

                # 处理可能的返回值类型
                if isinstance(result, tuple):
                    image_paths, prompt = result
                else:
                    image_paths = result
                    prompt = ""  # 设置默认提示文本

                for img_path in image_paths:
                    try:
                        # 保存 V 通道图像
                        v_channel_path = ImageProcessor.save_v_channel(img_path)

                        # 识别图像中的文本
                        results = ImageProcessor.process_image(v_channel_path, detector, recognizer)

                        print(f"🧠 {img_path.name} 默认识别结果: {results}")

                        if len(results) < 4:
                            print(f"❌ {img_path.name} 识别结果不足 4 个目标，准备刷新页面并重新开始识别。")
                            # crawler.refresh_page()
                            break

                        is_recognized = True
                        # 填充点击序列
                        click_sequence = fill_click_sequence(results, prompt)

                        print(f"🎯 {img_path.name} 点击序列: {click_sequence}")

                        # 执行点击
                        crawler._simulate_clicks(img_path, click_sequence)

                    except Exception as e:
                        print(f"❌ 处理 {img_path.name} 失败: {str(e)}")
