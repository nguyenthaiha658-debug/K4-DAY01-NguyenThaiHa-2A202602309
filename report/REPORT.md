# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
 {

    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K",
}
- Record này mô tả toàn ảnh như thế nào?
 record này mô tả ảnh là chiếc xe taxi
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
 người lập trình định nghĩa các class
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
 id để giúp máy sử dụng dễ dàng, tên lớp cho người hiểu, tên taxonomy là tên của từ điển class.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
 cần quy định về tiêu chí chọn chủ thể chính, nếu bị che khuất thì loại bỏ hay vẽ phần bị che khuất
- Vì sao model score không phải ground truth?
 model score là mức điểm mô hình đánh cho kết quả của mô hình, ground truth là kết quả do con người làm
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
{
    "class_name": "bus",
    "score": 0.912558,
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
},
- Diễn giải vị trí box bằng lời:
 vị trí dưới bên trái ở toạ độ (93.17, 187.95), trên bên phải ( 223.01, 320.91)

- So sánh số prediction ở hai threshold:
 threshold càng cao thì số prediction càng ít

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
 khi độ bao phủ tăng lên tức là nhiều vật thể có điểm thấp vẫn được lấy nên reviewer cần kiểm tra nhiều hơn
- Đề xuất một quy tắc box chặt:
 Bám sát biên vật thể: Khung xyxy phải ôm khít các pixel thực tế của đối tượng, tuyệt đối không để khoảng trống thừa chứa nền.

 Neo theo mép ảnh: Với vật thể nằm sát cạnh bức ảnh, mép hộp phải trùng khít với mép của bức ảnh.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
 Guideline phải ấn định rõ mức giới hạn (ví dụ: vật thể bị che khuất trên 70% thì bỏ qua không gán nhãn, dưới mức đó thì vẫn phải khoanh, Thống nhất việc chỉ khoanh phần diện tích thực tế còn hiển thị trong khung hình.).

 Quy trình Escalation: Khi gặp các trường hợp khó (vật thể nhập nhằng giữa nền và chủ thể, góc chụp gây ảo giác thị giác), reviewer không tự ý quyết định theo cảm tính mà phải đẩy lên cấp quản lý dự án (Lead/Manager) để thống nhất quy tắc chung và cập nhật vào guideline.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
{
    "instance_id": "traffic-001",
    "class_name": "bus",
    "score": 0.925745,
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],...
    ]
}
- Polygon bổ sung chi tiết gì so với box?
 Hộp giới hạn (xyxy) chỉ tạo khung hình chữ nhật thô, kéo theo rất nhiều khoảng trống nền ở các góc. Trong khi đó, polygon ôm sát từng đường cong và biên dạng thực tế của vật thể theo cấp độ pixel.

- `instance_id` dùng để làm gì và không phải loại ID nào?
 Dùng để làm gì: Phân biệt tường minh giữa các cá thể độc lập thuộc cùng một lớp trong một bức ảnh (ví dụ: phân biệt người A với người B khi cả hai đều thuộc lớp person).
 Không phải loại ID nào: Không phải là mã định danh lớp (class_id)

- Đề xuất một quy tắc biên mask:
 Bám sát ranh giới tương phản: Các đỉnh của đa giác phải nằm chính xác tại đường chuyển đổi độ sáng hoặc màu sắc giữa vật thể và nền, tuyệt đối không lấn chiếm vùng nền hoặc cắt cụt phần thực tế của đối tượng.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Quy định trong Guideline:
Vật thể dính liền nhau: Phải định rõ cách ngắt ranh giới giữa hai đối tượng cùng lớp dính sát vào nhau.
Vùng mờ/nhòe: Thống nhất điểm dừng của mặt nạ dựa trên điểm uốn của độ suy giảm tương phản.
Quy trình Escalation: Khi vùng biên quá nhập nhằng do điều kiện ánh sáng hoặc mức độ che khuất vượt quá ngưỡng cho phép (ví dụ che khuất trên 60%), reviewer không tự phỏng đoán mà phải chuyển ca đó lên cấp quản lý để quyết định đưa vào tập dữ liệu hay loại bỏ.



## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | `class_id`, `class_name`, `taxonomy_name` | Ảnh chứa nhiều chủ thể cạnh tranh, đối tượng quá nhỏ hoặc nhầm lẫn bối cảnh nền. | Xác định đối tượng trọng tâm dựa trên diện tích/tâm điểm, chọn nhãn chuẩn từ taxonomy. | Kiểm tra xem nhãn được chọn có phản ánh đúng đối tượng cốt lõi và tuân thủ guideline không. |
| Phát hiện vật thể | Tọa độ hộp `xyxy`, `class_id` | Hộp thừa nền (padding), bỏ sót vật thể nhỏ/mờ, hoặc bị cắt mép khung hình. | Vẽ khung chữ nhật ôm khít biên thực tế của đối tượng, gán đúng mã lớp. | Kiểm tra độ khít của bounding box, rà soát các đối tượng bị mô hình bỏ sót. |
| Instance segmentation | Đa giác/Mặt nạ (`polygon_points`), `instance_id`, `class_id` | Viền nhòe, vật thể cùng lớp dính sát nhau (touching), biên dạng phức tạp khó định hình. | Kéo điểm nối đa giác bám sát từng pixel đường viền, phân tách rõ ràng các `instance_id`. | Kiểm tra độ mượt và chính xác của biên mask, đảm bảo phân tách đúng các cá thể độc lập. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Nếu phát hiện hình ảnh hoặc dữ liệu không thuộc phạm vi cho phép (ngoài phạm vi xử lý, nhạy cảm, hoặc không đúng mục đích dự án), tôi sẽ dừng xử lý ngay lập tức
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Cấp quản lý trực tiếp

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
