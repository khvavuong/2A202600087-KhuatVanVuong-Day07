# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Khuất Văn Vương
**Nhóm:** Nhóm 69
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> Nghĩa là độ tương đồng giữa 2 vector có hướng là gần giống nhau trong vector store, tức là nội dung 2 câu mang ý nghĩa tương tự nhau nhưng độ dài có thể khác.

**Ví dụ HIGH similarity:**

- Sentence A: Tôi thích machine learning.
- Sentence B: Tôi thích tìm hiểu về học máy.
- Tại sao tương đồng: Cả 2 câu đều cùng một ý tưởng nhưng cách diẽn đạt và độ dài khác nhau.

**Ví dụ LOW similarity:**

- Sentence A: Tôi thích machine learning
- Sentence B: Hôm nay trời nắng
- Tại sao khác: Hai câu khác hẳn về ngữ nghĩa, hoàn cảnh, khác chủ đề

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> _Viết 1-2 câu:_ Vì cosine similarity thể hiện độ giống nhau về vector nên không bị ảnh hưởng bởi độ dài câu hay dim embedding. Text embedđing cần ưu tiên hơn về semantic, nên cosine similarity phù hợp hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> _Trình bày phép tính:_ (10000 - 50)/(500 - 50) = 23 chunks
> _Đáp án:_ 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> _Viết 1-2 câu:_ (10000 - 100) / (500 - 100) = 25 chunks. Overlap nhiều hơn thì tăng số chunks, tức là giảm độ trượt sau mỗi chunk, giúp giữ ngữ cảnh tốt hơn, giảm mất thông tin ở chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y tế sức khỏe

**Tại sao nhóm chọn domain này?**

> Khi tìm kiếm nguồn dữ liệu, nhận thấy dữ liệu y tế này có cấu trúc rõ ràng, đầy đủ phù hợp để thử nghiệm các chunking và metadata filtering. Nguồn dữ liệu từ phong phú, chất lượng cao, dễ crawl và preprocess sang markdown. Domain y tế cũng là một ứng dụng thực tiễn quan trọng của RAG, đòi hỏi retrieval chính xác theo chuyên khoa.

### Data Inventory

| #   | Tên tài liệu               | Nguồn             | Số ký tự | Metadata đã gán                                                            |
| --- | -------------------------- | ----------------- | -------- | -------------------------------------------------------------------------- |
| 1   | alzheimer.md               | Bệnh viện Tâm Anh | 24904    | disease_name: Bệnh Alzheimer, category: Thần kinh                          |
| 2   | an-khong-tieu.md           | Bệnh viện Tâm Anh | 10636    | disease_name: Ăn không tiêu, category: Tiêu hóa - Gan mật                  |
| 3   | ap-xe-hau-mon.md           | Bệnh viện Tâm Anh | 8115     | disease_name: Áp xe hậu môn, category: Tiêu hóa - Hậu môn trực tràng       |
| 4   | ap-xe-phoi.md              | Bệnh viện Tâm Anh | 12956    | disease_name: Áp xe phổi, category: Hô hấp                                 |
| 5   | ban-chan-dai-thao-duong.md | Bệnh viện Tâm Anh | 11761    | disease_name: Bàn chân đái tháo đường, category: Nội tiết - Đái tháo đường |
| 6   | bang-huyet-sau-sinh.md     | Bệnh viện Tâm Anh | 11496    | disease_name: Băng huyết sau sinh, category: Sản phụ khoa                  |
| 7   | bang-quang-tang-hoat.md    | Bệnh viện Tâm Anh | 9792     | disease_name: Bàng quang tăng hoạt, category: Tiết niệu                    |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị    | Tại sao hữu ích cho retrieval?                                               |
| --------------- | ------ | ---------------- | ---------------------------------------------------------------------------- |
| disease_name    | string | "Bệnh Alzheimer" | Filter theo tên bệnh, tránh nhầm lẫn giữa các bệnh có triệu chứng giống nhau |
| category        | string | "Thần kinh"      | Phân loại theo chuyên khoa, sử dụng để filter chính                          |
| source          | string | "BV Tâm Anh"     | Truy xuất nguồn gốc dữ liệu                                                  |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu               | Strategy     | Chunk Count | Avg Length | Preserves Context?             |
| ---------------------- | ------------ | ----------- | ---------- | ------------------------------ |
| Bệnh Alzheimer (24904) | fixed_size   | 55          | 494        | Thấp do bị chunk bỏ giữa chừng |
|                        | by_sentences | 74          | 366        | Cao                            |
|                        | recursive    | 73          | 371        | Cao                            |
| Ăn không tiêu (10636)  | fixed_size   | 25          | 487        | Thấp                           |
|                        | by_sentences | 41          | 295        | Cao                            |
|                        | recursive    | 33          | 368        | Cao                            |
| Áp xe hậu môn (8115)   | fixed_size   | 20          | 477        | Thấp                           |
|                        | by_sentences | 26          | 365        | Cao                            |
|                        | recursive    | 27          | 352        | Cao                            |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**

> _Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?_ RecursiveChunker thử lần lượt các separator theo thứ tự ưu tiên, theo paragraph, rồi dòng, rồi câu, rồi space, character-level, thực hiện các đệ quy. Các đoạn nhỏ liên tiếp được gộp lại cho đến khi gần chunk_size.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> _Viết 2-3 câu: domain có pattern gì mà strategy khai thác?_ Tài liệu y tế thu thập raw data bằng cách crawl từ web bệnh viện Tâm Anh dưới dạng .html, thực hiện preprocessing chuyển đổi có cấu trúc rõ ràng sang dạng markdown và paragraph. Sử dụng RecursiveChunker để khai thác cấu trúc này nhằm giữ nguyên các khối nội dung y tế có liên quan với nhau, tránh bị cắt ngang giữa chừng.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu               | Strategy                     | Chunk Count | Avg Length | Retrieval Quality?                               |
| ---------------------- | ---------------------------- | ----------- | ---------- | ------------------------------------------------ |
| Bệnh Alzheimer (24904) | best baseline (by_sentences) | 74          | 366        | Cao, nhưng chunk nhỏ lẻ, thiếu context liền mạch |
|                        | **RecursiveChunker (500)**   | 73          | 371        | Cao, giữ paragraph nguyên vẹn, gộp đoạn nhỏ      |
| Ăn không tiêu (10636)  | best baseline (by_sentences) | 41          | 295        | Nhiều chunk nhỏ, phân tán thông tin              |
|                        | **RecursiveChunker (500)**   | 33          | 368        | Ít chunk hơn, mỗi chunk chứa đủ ý                |

### So Sánh Với Thành Viên Khác

| Thành viên           | Strategy              | Embedding Model               | Vector DB | Precision (no filter) | Precision (filtered) |
| -------------------- | --------------------- | ----------------------------- | --------- | --------------------- | -------------------- |
| Khuất Văn Vương (me) | RecursiveChunker(500) | Qwen 0.8B                     | ChromaDB  | **95.2%**             | **100%**             |
| Nguyễn Đông Hưng     | RecursiveChunker(500) | OpenAI text-embedding-3-small | In-memory | **100%**              | **100%**             |
| Lưu Lương Vi Nhân    | Recursive(400)        | all-MiniLM-L6-v2              | ChromaDB  | **66.8%**             | **100%**             |
| Huỳnh Văn Nghĩa      | SentenceChunker(500)  | GPT-4o-mini                   | ChromaDB  | **9.5%**              | **100%**             |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> RecursiveChunker kết hợp qwen embedding và metadata filter theo category cho kết quả tốt nhất. RecursiveChunker giữ nguyên cấu trúc paragraph/heading của tài liệu y tế markdown, trong khi metadata filter theo chuyên khoa giúp loại bỏ noise từ các bệnh khác category, nâng precision từ 95.2% lên 100%.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Sử dụng regex `(?<=[.!?])\s+` để detect ranh giới câu (lookbehind cho dấu `.`, `!`, `?` theo sau bởi whitespace). Xử lý edge case: text rỗng trả về `[]`, strip whitespace mỗi câu, bỏ qua câu rỗng sau khi split. Gộp các câu thành chunk theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Algorithm hoạt động đệ quy: thử split text bằng separator ưu tiên cao nhất (`\n\n` → `\n` → `. ` → ` ` → `""`). Nếu separator không tách được (chỉ 1 phần), chuyển sang separator tiếp theo. Các phần nhỏ liên tiếp được gộp lại cho đến khi vượt `chunk_size`, phần quá lớn tiếp tục đệ quy với separator kế. Base case: text đã nhỏ hơn `chunk_size` thì trả về nguyên, hoặc hết separator thì cắt cứng từng `chunk_size` ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> Lưu trữ qua ChromaDB (primary) hoặc in-memory fallback. Mỗi document được embed bằng `embedding_fn`, tạo record gồm id, content, embedding, metadata rồi `collection.add()` vào ChromaDB. Khi search, gọi `collection.query()` với query embedding, ChromaDB trả về distances (L2), convert sang similarity score bằng công thức `score = 1.0 / (1.0 + distance)` rồi sort descending.

**`search_with_filter` + `delete_document`** — approach:

> Filter **trước** khi search: truyền `where=metadata_filter` vào `collection.query()` để ChromaDB lọc metadata trước rồi mới tính similarity trên tập đã lọc. Với in-memory, filter bằng cách duyệt `self._store` và so khớp metadata trước, rồi mới gọi `_search_records()`. Delete bằng cách so sánh `collection.count()` trước và sau khi gọi `collection.delete(where={"doc_id": doc_id})`, trả về `True` nếu count giảm.

### KnowledgeBaseAgent

**`answer`** — approach:

> Prompt structure gồm 3 phần: system instruction ("Use ONLY the context below"), context (các chunk retrieved ghép bằng `\n\n`), và question. Cách inject context: gọi `store.search()` (hoặc `search_with_filter()` nếu có `metadata_filter`) để lấy top-k chunks, nối content thành chuỗi context, nhúng vào template prompt rồi gọi `llm_fn()`. Nếu không tìm được chunk nào, trả về thông báo mặc định thay vì gọi LLM.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
============== 42 passed ==============
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                                    | Sentence B                                    | Dự đoán | Actual Score | Đúng? |
| ---- | --------------------------------------------- | --------------------------------------------- | ------- | ------------ | ----- |
| 1    | Bệnh Alzheimer gây suy giảm trí nhớ           | Mất trí nhớ do thoái hóa thần kinh            | high    | high         | Đúng  |
| 2    | Triệu chứng áp xe phổi là sốt và ho           | Áp xe phổi biểu hiện bằng sốt cao, ho có đờm  | high    | high         | Đúng  |
| 3    | Bàn chân đái tháo đường cần chăm sóc đặc biệt | Hôm nay thời tiết đẹp quá                     | low     | low          | Đúng  |
| 4    | Băng huyết sau sinh là biến chứng nguy hiểm   | Bàng quang tăng hoạt gây tiểu không kiểm soát | low     | low          | Đúng  |
| 5    | Ăn không tiêu do rối loạn tiêu hóa            | Áp xe hậu môn là bệnh về hậu môn trực tràng   | low     | medium       | X     |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Pair 5 cho thấy dù "ăn không tiêu" và "áp xe hậu môn" thuộc hai bệnh khác nhau, embedding cho similarity medium vì cả hai đều nằm trong lĩnh vực tiêu hóa và chia sẻ từ vựng y tế tương tự. Điều này cho thấy embeddings không chỉ so khớp từ khóa mà còn nắm bắt semantic field, các câu cùng domain y tế tiêu hóa sẽ gần nhau trong vector space dù nói về bệnh khác nhau.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                     | Gold Answer                                                                                     |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | Bệnh Alzheimer có di truyền không?        | Có yếu tố di truyền, đặc biệt gen APOE-e4, nhưng không phải tất cả người mang gen đều mắc bệnh  |
| 2   | Nguyên nhân gây ăn không tiêu là gì?      | Do rối loạn chức năng tiêu hóa, stress, ăn quá nhanh, thức ăn nhiều dầu mỡ, hoặc bệnh lý dạ dày |
| 3   | Áp xe hậu môn có tự khỏi không?           | Không tự khỏi, cần phẫu thuật dẫn lưu mủ, nếu không điều trị có thể biến chứng thành rò hậu môn |
| 4   | Triệu chứng của áp xe phổi?               | Sốt cao, ho có đờm mủ hôi, đau ngực, khó thở, sụt cân                                           |
| 5   | Bàn chân đái tháo đường chăm sóc thế nào? | Kiểm tra chân hàng ngày, giữ vệ sinh, mang giày phù hợp, kiểm soát đường huyết, khám định kỳ    |

### Kết Quả Của Tôi

| #   | Query                                     | Top-1 Retrieved Chunk (tóm tắt)                                               | Score | Relevant? | Agent Answer (tóm tắt)                                                                       |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------- | ----- | --------- | -------------------------------------------------------------------------------------------- |
| 1   | Bệnh Alzheimer có di truyền không?        | Chunk về yếu tố nguy cơ Alzheimer, đề cập gen APOE-e4 và tiền sử gia đình     | 0.62  | Đúng      | Có yếu tố di truyền, gen APOE-e4 tăng nguy cơ, nhưng không phải nguyên nhân duy nhất         |
| 2   | Nguyên nhân gây ăn không tiêu là gì?      | Chunk về nguyên nhân ăn không tiêu: rối loạn tiêu hóa, stress, thức ăn dầu mỡ | 0.58  | Đúng      | Do nhiều nguyên nhân: rối loạn chức năng dạ dày, stress, ăn quá nhanh, thức ăn nhiều dầu mỡ  |
| 3   | Áp xe hậu môn có tự khỏi không?           | Chunk về điều trị áp xe hậu môn, phẫu thuật dẫn lưu mủ                        | 0.60  | Đúng      | Không tự khỏi, cần can thiệp phẫu thuật dẫn lưu, nếu không điều trị sẽ biến chứng rò hậu môn |
| 4   | Triệu chứng của áp xe phổi?               | Chunk về biểu hiện lâm sàng: sốt cao, ho đờm mủ, đau ngực                     | 0.61  | Đúng      | Sốt cao kéo dài, ho ra đờm mủ có mùi hôi, đau ngực, khó thở, mệt mỏi                         |
| 5   | Bàn chân đái tháo đường chăm sóc thế nào? | Chunk về chăm sóc bàn chân: kiểm tra hàng ngày, vệ sinh, giày phù hợp         | 0.59  | Đúng      | Kiểm tra bàn chân mỗi ngày, giữ vệ sinh sạch sẽ, mang giày vừa vặn, kiểm soát đường huyết    |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> OpenAI text-embedding-3-small cho precision 100% ngay cả khi không filter, chứng tỏ chất lượng embedding model ảnh hưởng rất lớn đến retrieval accuracy. Tuy nhiên, qwen 0.6B là model local miễn phí và vẫn đạt 95.2% precision, cho thấy trade-off giữa chi phí và hiệu suất là có thể chấp nhận được.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> Thấy được tầm quan trọng của preprocessing dữ liệu — việc chuyển HTML thô sang markdown có cấu trúc giúp chunking hiệu quả hơn nhiều. Nhóm khác sử dụng overlap chunking cũng cho kết quả thú vị trong việc giữ context liên tục giữa các chunk.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Sẽ thêm overlap cho RecursiveChunker để tránh mất context ở biên chunk. Ngoài ra, sẽ thử fine-tune thêm metadata schema — ví dụ thêm trường `section` (triệu chứng/nguyên nhân/điều trị) để filter chính xác hơn theo loại thông tin cần tìm, không chỉ theo chuyên khoa. Cũng sẽ thử hybrid search (BM25 + vector) để cải thiện recall cho các query chứa thuật ngữ y tế chuyên biệt.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5 / 5            |
| Document selection          | Nhóm    | 9 / 10           |
| Chunking strategy           | Nhóm    | 15 / 15          |
| My approach                 | Cá nhân | 10 / 10          |
| Similarity predictions      | Cá nhân | 4 / 5            |
| Results                     | Cá nhân | 9 / 10           |
| Core implementation (tests) | Cá nhân | 30 / 30          |
| Demo                        | Nhóm    | 5 / 5            |
| **Tổng**                    |         | **87 / 100**     |
