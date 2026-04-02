import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 尝试导入 OpenAI（如果安装了的话）---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告：未安装 openai 库，AI 总结功能将不可用")
    print("如需使用 AI 总结，请运行: pip install openai")

# --- 配置区 ---
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
SERVERCHAN_KEY = os.getenv("SERVERCHAN_KEY")

HISTORY_FILE = "config/seen_papers.txt"
DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def create_session():
    session = requests.Session()
    # 配置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if OPENALEX_API_KEY:
        session.headers.update({"Authorization": f"Bearer {OPENALEX_API_KEY}"})
        debug_print("使用 API Key 调用 OpenAlex")
    else:
        debug_print("未设置 OpenAlex API Key，使用无 Key 调用")
    session.headers.update({"mailto": "your-email@example.com"})
    return session

def read_list(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_seed_papers(file_path):
    papers = []
    if not os.path.exists(file_path):
        debug_print(f"Warning: 找不到文件 {file_path}")
        return papers
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                papers.append(line)
    return papers

def fetch_work_by_id(session, paper_input):
    """
    获取论文详情，支持 DOI 或 OpenAlex ID。
    """
    # 判断输入类型
    if paper_input.startswith("10."):
        url = f"https://api.openalex.org/works/doi/{paper_input}"
    elif paper_input.startswith("W"):
        url = f"https://api.openalex.org/works/{paper_input}"
    elif paper_input.startswith("https://doi.org/"):
        doi = paper_input.replace("https://doi.org/", "")
        url = f"https://api.openalex.org/works/doi/{doi}"
    elif paper_input.startswith("https://openalex.org/"):
        url = paper_input
    else:
        url = f"https://api.openalex.org/works/doi/{paper_input}"

    # 有效字段列表（移除 host_venue）
    params = {
        "select": "id,title,abstract_inverted_index,doi,publication_date,publication_year,primary_location,cited_by_count,related_works"
    }
    debug_print(f"请求 URL: {url}")
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            debug_print(f"获取论文失败，状态码 {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except Exception as e:
        debug_print(f"请求异常: {e}")
        return None

def fetch_works_batch(session, work_ids):
    if not work_ids:
        return []
    ids_str = "|".join([wid.split("/")[-1] for wid in work_ids])
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"openalex_id:{ids_str}",
        "per-page": len(work_ids),
        "select": "id,title,abstract_inverted_index,doi,publication_date,publication_year,primary_location,cited_by_count,related_works"
    }
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            debug_print(f"批量查询失败，状态码 {resp.status_code}: {resp.text[:200]}")
            return []
        return resp.json().get("results", [])
    except Exception as e:
        debug_print(f"批量查询异常: {e}")
        return []

def find_by_related_works(session, seed_work, limit=10):
    debug_print("正在通过 related_works 查找相关文献...")
    related_ids = seed_work.get("related_works", [])
    debug_print(f"找到 {len(related_ids)} 篇相关文献 ID")
    if not related_ids:
        return []
    related_ids = related_ids[:limit]
    papers = fetch_works_batch(session, related_ids)
    debug_print(f"成功获取 {len(papers)} 篇 related_works 论文")
    return papers

def find_by_semantic_search(session, seed_work, limit=10):
    debug_print("正在通过语义搜索查找相关文献...")
    search_text = seed_work.get("abstract_inverted_index") and " " or seed_work.get("title", "")
    if not search_text:
        search_text = seed_work.get("title", "")
    search_text = search_text[:500].strip()
    if not search_text:
        return []
    url = "https://api.openalex.org/works"
    params = {
        "search": search_text,
        "per-page": limit,
        "sort": "relevance",
        "select": "id,title,abstract_inverted_index,doi,publication_date,publication_year,primary_location,cited_by_count,related_works"
    }
    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            debug_print(f"语义搜索失败，状态码 {resp.status_code}: {resp.text[:200]}")
            return []
        results = resp.json().get("results", [])
        seed_id = seed_work.get("id", "")
        filtered = [p for p in results if p.get("id") != seed_id]
        return filtered[:limit]
    except Exception as e:
        debug_print(f"语义搜索异常: {e}")
        return []

def undo_inverted_index(inverted_index):
    """将 OpenAlex 的倒排索引还原为原始文本"""
    if not inverted_index:
        return ""
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append([word, pos])
    word_index.sort(key=lambda x: x[1])
    words = [pair[0] for pair in word_index]
    return ' '.join(words)

def fetch_abstract_from_pubmed(doi):
    """通过 PubMed API 获取论文摘要（仅当 DOI 有效时）"""
    if not doi:
        return ""
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={doi}[DOI]&retmode=json"
    try:
        resp = requests.get(search_url, timeout=10)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return ""
        pmid = pmids[0]
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
        resp = requests.get(fetch_url, timeout=10)
        if resp.status_code != 200:
            return ""
        root = ET.fromstring(resp.content)
        abstract_elem = root.find(".//Abstract")
        if abstract_elem is None:
            return ""
        abstract_parts = []
        for text_elem in abstract_elem.findall("AbstractText"):
            if text_elem.text:
                abstract_parts.append(text_elem.text.strip())
        return " ".join(abstract_parts)
    except Exception as e:
        debug_print(f"PubMed 摘要获取失败: {e}")
        return ""

def deduplicate_papers(papers):
    seen = set()
    unique = []
    for p in papers:
        pid = p.get("id", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)
    debug_print(f"去重前: {len(papers)} 篇，去重后: {len(unique)} 篇")
    return unique

def filter_unseen_papers(papers):
    seen = set(read_list(HISTORY_FILE))
    unseen = []
    for p in papers:
        pid = p.get("paperId", "")
        if pid and pid not in seen:
            unseen.append(p)
    debug_print(f"过滤后剩余 {len(unseen)} 篇未读论文")
    return unseen

def convert_to_standard_format(openalex_paper):
    """将 OpenAlex 论文格式转换为标准格式，并尝试补全摘要"""
    # 提取作者信息
    authors = openalex_paper.get("authorships", [])
    author_list = []
    for auth in authors[:5]:
        author = auth.get("author", {})
        author_list.append({"name": author.get("display_name", "未知")})
    # 提取 DOI
    doi = openalex_paper.get("doi", "")
    if doi:
        doi = doi.replace("https://doi.org/", "")
    # 提取年份
    pub_date = openalex_paper.get("publication_date", "")
    year = pub_date[:4] if pub_date else openalex_paper.get("publication_year", "")
    # 提取会议/期刊
    venue = ""
    primary_loc = openalex_paper.get("primary_location", {})
    source = primary_loc.get("source", {})
    if source:
        venue = source.get("display_name", "")
    # 获取摘要：优先使用倒排索引还原，若为空则尝试 PubMed
    abstract = ""
    inverted = openalex_paper.get("abstract_inverted_index")
    if inverted:
        abstract = undo_inverted_index(inverted)
    if not abstract and doi:
        abstract = fetch_abstract_from_pubmed(doi)
    return {
        "paperId": openalex_paper["id"].split("/")[-1],
        "title": openalex_paper.get("title", "无标题"),
        "abstract": abstract,
        "authors": author_list,
        "url": openalex_paper.get("doi", f"https://openalex.org/works/{openalex_paper['id']}"),
        "venue": venue,
        "externalIds": {"DOI": doi} if doi else {},
        "publicationDate": pub_date,
        "year": year,
        "cited_by_count": openalex_paper.get("cited_by_count", 0)
    }

def sort_by_cited_count(papers):
    return sorted(papers, key=lambda x: x.get("cited_by_count", 0), reverse=True)

def get_paper_recommendations():
    seed_papers = read_seed_papers("config/seed_paper_positive.csv")
    if not seed_papers:
        print("错误：至少需要一篇 Positive 论文作为基准")
        return []
    session = create_session()
    all_papers = []
    for seed_input in seed_papers[:1]:
        print(f"\n处理种子论文: {seed_input}")
        seed_work = fetch_work_by_id(session, seed_input)
        if not seed_work:
            print(f"无法获取种子论文: {seed_input}")
            continue
        print(f"种子论文标题: {seed_work.get('title', 'N/A')}")
        print(f"引用数: {seed_work.get('cited_by_count', 0)}")
        related = find_by_related_works(session, seed_work, limit=10)
        semantic = find_by_semantic_search(session, seed_work, limit=10)
        all_papers = related + semantic
        print(f"合并后共 {len(all_papers)} 篇")
    if not all_papers:
        print("未找到任何推荐论文")
        return []
    unique = deduplicate_papers(all_papers)
    standard = [convert_to_standard_format(p) for p in unique]
    sorted_papers = sort_by_cited_count(standard)
    unseen = filter_unseen_papers(sorted_papers)
    final = unseen[:15]
    print(f"\n最终推荐 {len(final)} 篇论文")
    return final

def format_papers_simple(papers):
    report = ""
    for idx, paper in enumerate(papers, 1):
        title = paper.get("title", "无标题")
        date = paper.get("publicationDate") or paper.get("year") or "未知日期"
        abstract = (paper.get("abstract") or "").strip()
        preview = abstract[:500] + "..." if len(abstract) > 500 else abstract
        doi = paper.get("externalIds", {}).get("DOI", "")
        url = paper.get("url", "")
        if doi and not url:
            url = f"https://doi.org/{doi}"
        if not url:
            url = f"https://openalex.org/works/{paper.get('paperId', '')}"
        venue = (paper.get("venue") or "").strip() or "未知"
        cited = paper.get("cited_by_count", 0)
        authors = paper.get("authors", [])
        if len(authors) > 4:
            names = [authors[0].get("name","未知"), authors[1].get("name","未知"),
                     "...", authors[-2].get("name","未知"), authors[-1].get("name","未知")]
            author_str = ", ".join(names)
        else:
            author_str = ", ".join([a.get("name","未知") for a in authors])
        report += (
            f"## {idx}\n"
            f"[{title}]({url})\n\n"
            f"*{venue}* | {author_str} | {date} | 📊 被引: {cited}\n\n"
            f"**摘要预览:**\n{preview}\n\n"
            f"---\n\n"
        )
    return report

def summarize_papers_with_llm(papers):
    if not LLM_API_KEY:
        print("未设置 LLM_API_KEY，使用简单格式推送...")
        return format_papers_simple(papers)
    if not OPENAI_AVAILABLE:
        print("openai 库未安装，回退到简单格式...")
        return format_papers_simple(papers)
    print("正在使用 AI 总结论文...")
    client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")
    report = ""
    for idx, paper in enumerate(papers, 1):
        title = paper.get("title", "无标题")
        date = paper.get("publicationDate") or paper.get("year") or "未知日期"
        abstract = (paper.get("abstract") or "").strip() or "无摘要"
        doi = paper.get("externalIds", {}).get("DOI", "")
        url = paper.get("url", "")
        if doi and not url:
            url = f"https://doi.org/{doi}"
        if not url:
            url = f"https://openalex.org/works/{paper.get('paperId', '')}"
        venue = (paper.get("venue") or "").strip() or "未知"
        cited = paper.get("cited_by_count", 0)
        authors = paper.get("authors", [])
        if len(authors) > 4:
            names = [authors[0].get("name","未知"), authors[1].get("name","未知"),
                     "...", authors[-2].get("name","未知"), authors[-1].get("name","未知")]
            author_str = ", ".join(names)
        else:
            author_str = ", ".join([a.get("name","未知") for a in authors])
        prompt = f"""
你是一个严谨的学术专家。请基于以下论文信息，提取核心内容并转化为中文。
要求：
1. 极其精简、具体，拒绝空泛的套话，保留专业术语。
2. 绝对不要输出任何诸如"好的，这是为您总结的论文"之类的客套话。
3. 请严格按照以下 Markdown 格式输出:
**[试图解决的问题]**：(用一句话概括该研究针对的痛点或背景)
**[核心方法]**：(具体使用了什么架构、算法、模型或机制)
**[创新与效果]**：(实现了什么指标提升，或解决了什么具体的限制)

标题: {title}
摘要原文: {abstract}
"""
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            summary = resp.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败 ({title[:30]}...): {e}")
            summary = "（AI 总结生成失败，请查看原文）"
        report += (
            f"## {idx}\n"
            f"[{title}]({url})\n"
            f"*{venue}* | {author_str} | {date} | 📊 被引: {cited}\n\n"
            f"{summary}\n\n"
            f"---\n\n"
        )
    return report

def update_history(papers):
    if not papers:
        return
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        for p in papers:
            f.write(p.get("paperId") + "\n")
    print(f"已更新历史记录，新增 {len(papers)} 篇")

def push_to_wechat(content):
    if not SERVERCHAN_KEY:
        print("错误：未设置 SERVERCHAN_KEY，无法推送")
        return False
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
    data = {"title": "📚 你的每日文献追踪晨报到了！", "desp": content}
    try:
        r = requests.post(url, data=data, timeout=30)
        if r.status_code == 200:
            print("推送成功！")
            return True
        else:
            print(f"推送失败: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"推送异常: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("论文追踪系统 (基于 OpenAlex + PubMed 摘要补全)")
    print("=" * 50)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("配置状态:")
    print(f"  - OpenAlex API Key: {'已设置' if OPENALEX_API_KEY else '未设置'}")
    print(f"  - LLM API Key: {'已设置（将启用 AI 总结）' if LLM_API_KEY else '未设置（简单格式）'}")
    print(f"  - openai 库: {'已安装' if OPENAI_AVAILABLE else '未安装'}")
    print(f"  - Server酱 Key: {'已设置' if SERVERCHAN_KEY else '未设置'}\n")
    print("正在寻找最新推荐...")
    new_papers = get_paper_recommendations()
    if new_papers:
        print(f"\n找到 {len(new_papers)} 篇最新论文")
        print("正在生成报告...")
        report = summarize_papers_with_llm(new_papers)
        print("正在推送到微信...")
        if push_to_wechat(report):
            print("更新历史记录...")
            update_history(new_papers)
            print("全部完成！")
        else:
            print("推送失败，不更新历史记录")
    else:
        print("今天没有发现未读的最新相关文献。")
