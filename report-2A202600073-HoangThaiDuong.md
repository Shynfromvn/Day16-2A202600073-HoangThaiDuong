# Báo Cáo Ngắn Phương Án CPU
# Hoàng Thái Dương - 2A202600073

Do tài khoản AWS không được cấp quota GPU, em chuyển sang phương án CPU theo Phần 7 trong `README_aws.md` để vẫn hoàn thành đầy đủ quy trình Terraform IaC, SSH vào cloud instance, tải dataset, train model và kiểm tra chi phí.
Hạ tầng AWS đã được khởi tạo thành công, truy cập vào bastion và CPU node thành công, sau đó chạy benchmark LightGBM trên bộ dữ liệu Credit Card Fraud Detection.
Kết quả benchmark cho thấy thời gian load dữ liệu là 1.0203 giây và thời gian training là 6.6639 giây, cho thấy CPU node vẫn đủ mạnh để xử lý bài toán machine learning thực tế.
Model đạt AUC-ROC 0.977448 và Accuracy 0.999579, cho thấy mô hình học tốt và phân loại rất chính xác trên tập test.
Chỉ số F1-Score đạt 0.875, Precision đạt 0.893617 và Recall đạt 0.857143, phù hợp với bài toán fraud detection khi cần cân bằng giữa phát hiện đúng và hạn chế báo động giả.
Về hiệu năng suy luận, độ trễ cho 1 bản ghi là 0.641864 ms và throughput cho 1000 bản ghi đạt 79797.55679 rows/sec, cho thấy khả năng inference nhanh trên CPU.
Phương án CPU không cần xin quota đặc biệt như GPU, dễ triển khai hơn trên tài khoản mới và vẫn đáp ứng yêu cầu benchmark, training và báo cáo kết quả của bài lab.
Sau khi thu thập đầy đủ ảnh benchmark, file JSON kết quả và thông tin billing, toàn bộ tài nguyên đã được hủy bằng `terraform destroy` để tránh phát sinh chi phí.
