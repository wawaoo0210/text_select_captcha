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
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# 常量定义
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_CHANNEL_DIR = Path("channel")
RAW_IMAGE_DIR = Path("./images_raw")
BASE64_PREFIX = "data:image/png;base64,"
BORDER_SHRINK = 5
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
        """使用预存坐标进行点击"""
        img_element = self.driver.find_element(By.XPATH, '//img')
        img_rect = self.driver.execute_script(
            "return arguments[0].getBoundingClientRect();", img_element)

        # 读取原始图像尺寸
        img = cv2.imread(str(image_path))
        original_h, original_w = img.shape[:2]

        # 计算缩放比例
        scale_x = img_rect['width'] / original_w
        scale_y = img_rect['height'] / original_h

        actions = ActionChains(self.driver)
        for text, bbox in click_sequence:
            if not bbox:
                print(f"❌ 缺失点击坐标: {text}")
                continue

            x1, y1, x2, y2 = bbox
            # 计算中心点并转换坐标
            center_x = (x1 + x2) // 2 * scale_x
            center_y = (y1 + y2) // 2 * scale_y

            # 使用相对坐标点击
            actions.move_to_element_with_offset(
                img_element, center_x, center_y).click().perform()
            print(f"✅ 点击 {text} 坐标: ({center_x:.1f}, {center_y:.1f})")
            time.sleep(random.uniform(0.3, 0.7))


def fill_click_sequence(results, prompt):
    click_sequence = []
    for char in prompt:
        found = next(((t, b) for t, b in results if t == char), None)
        click_sequence.append(found or (char, None))

    # 补充自定义模型识别
    if any(b is None for _, b in click_sequence):
        alt_results = ImageProcessor.process_image(v_channel_path, detector, myocr)
        for i, (char, _) in enumerate(click_sequence):
            if not click_sequence[i][1]:
                match = next((b for t, b in alt_results if t == char), None)
                if match:
                    click_sequence[i] = (char, match)

    # 随机填充空白项
    for i, (char, bbox) in enumerate(click_sequence):
        if bbox is None:
            click_sequence[i] = (char, (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))

    return click_sequence


if __name__ == "__main__":
    if not os.path.exists("images_raw"):
        os.makedirs("images_raw")
    if not os.path.exists("channel"):
        os.makedirs("channel")
    if not os.path.exists("output"):
        os.makedirs("output")

    with WebCrawler() as crawler:
        result = crawler.crawl_images()

        # 处理可能的返回值类型
        if isinstance(result, tuple):
            image_paths, prompt = result
        else:
            image_paths = result
            prompt = ""  # 设置默认提示文本

        detector = ddddocr.DdddOcr(det=True)
        recognizer = ddddocr.DdddOcr()

        myocr = ddddocr.DdddOcr(det=True,
                                import_onnx_path="models/click_captcha_0.65625_474_38000_2025-04-16-15-14-11.onnx",
                                charsets_path="models/charsets.json")

        for img_path in image_paths:
            try:
                # 保存 V 通道图像
                v_channel_path = ImageProcessor.save_v_channel(img_path)

                # 识别图像中的文本
                results = ImageProcessor.process_image(v_channel_path, detector, recognizer)

                print(f"🧠 {img_path.name} 默认识别结果: {results}")

                # 填充点击序列
                click_sequence = fill_click_sequence(results, prompt)

                print(f"🎯 {img_path.name} 点击序列: {click_sequence}")

                # 执行点击
                crawler._simulate_clicks(img_path, click_sequence)

            except Exception as e:
                print(f"❌ 处理 {img_path.name} 失败: {str(e)}")
