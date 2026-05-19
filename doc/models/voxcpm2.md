# VoxCPM2 系列能力说明

[官方最佳实践文档](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html)
本文档面向希望接入 StreamVox 的开发者，集中说明 `VoxCPM2` 在当前 `voxcpm2-gguf` 接入形态下的能力边界、推荐用法与最佳实践。  
你可以把它当作 [Usage Guide](../usage.md) 的补充：`usage.md` 侧重 SDK 入口和通用参数，这一页更侧重 **VoxCPM2 适合做什么、当前 StreamVox 具体开放了什么、以及怎样写更容易出效果**。

`voxcpm2-gguf` 同时覆盖三类非常核心的能力：**音色设计、非语言控制、音色克隆**。这意味着你既可以在**没有参考音频**时，仅通过 `control_text` 直接设计一个目标声音；也可以在正文里加入 `[laughing]`、`[sigh]`、`[Shh]` 这类**非语言标签**，让语气、停顿和表现状态更自然；还可以基于**参考音频**去复现目标音色，并在克隆后的声音基础上继续调整表达方式。对实际业务来说，这让 VoxCPM2 不只是“把文本念出来”，而是能同时覆盖**声音设定、表现控制和角色复用**三类常见需求。



## 1. 模型能力

### 1.1 四种生成模式

`voxcpm2-gguf` 当前公开支持以下四种模式：

| 模式名 | 数字模式 | 是否需要参考音频 | 是否需要 `prompt_text` | `control_text` 是否生效 | 典型用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `text` | `0` | 否 | 否 | 是 | 纯文本直接生成，适合 Voice Design。 |
| `ref` | `1` | 是 | 否 | 是 | 参考音色克隆，并可用 `control_text` 微调表达。 |
| `continuation` | `2` | 是 | 是 | 否 | 基于一段提示音频和对应文本做续写。 |
| `ref_continuation` | `3` | 是 | 是 | 否 | 同时带参考音色与续写上下文的生成。 |

需要特别注意：

* `control_text` **只在** `text` 和 `ref` 模式下生效。
* 当模式是 `continuation` 或 `ref_continuation` 时，当前实现会忽略 `control_text`。
* 如果你真正想做的是“根据文字描述直接设计声音”或“在克隆基础上改语速、情绪、表达方式”，应优先选择 `text` 或 `ref`，而不是 continuation 系模式。

### 1.2 多语种与文本覆盖

结合当前 `voxcpm2-gguf` 的文本预处理规则，StreamVox 已为 VoxCPM2 保留了较完整的多语种正文范围。当前实现覆盖的代表性语言包括：

* 亚洲：中文、日语、韩语、印地语、泰语、越南语、印尼语、马来语、他加禄语、高棉语、老挝语、缅甸语
* 欧洲与美洲：英语、法语、德语、西班牙语、意大利语、俄语、葡萄牙语、荷兰语、波兰语、瑞典语、丹麦语、芬兰语、挪威语、希腊语
* 中东与非洲：阿拉伯语、土耳其语、希伯来语、斯瓦希里语

对 `voxcpm2-gguf` 来说，一个很实用的经验是：

* **大多数情况下直接写目标语言正文即可**
* 当前模型私有参数里**不依赖单独的语言模式开关**来切换这几类能力
* 比起堆额外标识，先把正文写干净、写自然，通常更重要

### 1.3 方言与非语言标签

在当前实现里，以下特殊标签会被文本预处理明确保护，不会在清洗和分段阶段被拆坏：

* `[laughing]`
* `[sigh]`
* `[Uhm]`
* `[Shh]`
* `[Question-ah]`
* `[Question-ei]`
* `[Question-en]`
* `[Question-oh]`
* `[Surprise-wa]`
* `[Surprise-yo]`
* `[Dissatisfaction-hnn]`

这也是为什么对于 VoxCPM2，推荐优先使用**英文方括号标签**。  
这类标签在当前 `voxcpm2-gguf` 文本链路里是“显式照顾过”的，稳定性通常会比随手发明的新标签更好。

方言方面，当前实现不会替你把普通话自动改写成地道方言，因此：

* 如果你要粤语、四川话、东北话、河南话这类地方表达，**最好直接写当地常见说法**
* 不要指望“普通话正文 + 一个方言名”总能稳定得到地道方言
* 如果你不会写方言，先用文本助手把正文改写成更自然的方言版本，往往比硬调参数更有效

## 2. 最佳实践

### 2.1 先把正文写对，再谈控制

对 VoxCPM2 来说，最有效的优化通常不是先堆控制词，而是先把正文写自然。

推荐顺序：

1. 先写干净的目标语言正文
2. 再决定是否要加方言表达
3. 再决定是否要加非语言标签
4. 最后才补 `control_text` 或切换模式

如果正文本身就拧巴、机械或者不符合口语习惯，再强的模型也很难把它“救”成理想效果。

### 2.2 做 Voice Design 时，优先用 `text`

如果你没有参考音频，只是想通过自然语言直接设计一个声音，推荐优先用 `text` 模式。

`control_text` 的推荐写法，是把以下几类信息揉成**一条完整描述**：

* 身份设定：例如性别、年龄、职业、人物气质
* 声音质感：例如低沉、沙哑、明亮、磁性
* 表达状态：例如激昂、低声、缓慢、旁白感
* 场景意图：例如历史讲述、新闻播报、口号宣讲

示例方向：

```text
热情洋溢的中年男性播音员，声音较为低沉，富有磁性与感染力，带着逐渐密集的节奏感呼喊宣讲口号。
```

```text
A quiet raspy, elderly woman of a low-pitched voice with a distinct, grainy texture and subtle breathy tremors. Delivers a slow tone at a very low volume, perfect for historical narration.
```

### 2.3 做音色克隆时，优先用 `ref`

如果你已经有目标音色的参考音频，而你真正想做的是：

* 复现这个人的音色
* 在保留这个人声音底色的同时微调表达
* 让同一个角色说新的内容

那么推荐优先使用 `ref` 模式，而不是直接上 continuation。

原因很简单：

* `ref` 模式下 `control_text` 仍然有效
* 你可以在保留音色的同时，继续控制语速、情绪和表达方式
* 对很多“角色克隆 + 风格微调”的业务场景来说，这比 continuation 更直接

实用建议：

* 参考音频最好 **5 秒以上**
* 音频越干净、越稳定，克隆效果通常越好
* 如果想要更快、更亮、更有情绪，可以在 `ref` 模式下再加一条简洁的 `control_text`

### 2.4 做续写时，别把 `continuation` 当成加强版 `ref`

`continuation` 和 `ref_continuation` 的核心价值，不是“更强克隆”，而是**更强的上下文续接**。

适合使用 continuation 的情况：

* 你有一段已经说出来的音频
* 你知道这段音频对应的 `prompt_text`
* 你希望模型在这个上下文上继续往后说

这时最重要的不是 `control_text`，而是：

* 提示音频本身是否干净
* `prompt_text` 是否与提示音频严格一致
* 待合成文本是否真的属于“往后接着说”的关系

如果你只是想“拿这个人的音色说一段新话”，多数情况下 `ref` 更合适。  
如果你想“沿着这一段已经说出来的话继续往后接”，才优先考虑 `continuation`。

### 2.5 `ref_continuation` 的正确理解

当前公开的 `ref_continuation` 不是“上传两条不同音频，一条做参考、一条做续写”的接口。  
在当前 StreamVox 入口下，它更准确的理解应该是：

* 用**同一条**音频同时提供音色参考
* 同时提供 continuation 所需的上下文线索
* 再配合 `prompt_text` 做连续生成

所以它最适合：

* 你只有一条高质量单人音频
* 既想保留这个人的音色，又想延续这段音频的上下文状态
* 不依赖 `control_text`

### 2.6 标签要少而准

虽然 VoxCPM2 支持非语言标签，但不建议在一句话里叠太多标签。  
对当前 `voxcpm2-gguf`，更稳妥的做法是：

* 优先使用当前已被保护的标签集合
* 一句里点到为止
* 真正需要笑声、停顿、迟疑时再加
* 不要把标签当成主控制手段

推荐方向：

```text
[Shh] 先别出声，他们已经到门口了。
```

```text
[laughing] 你刚才那个反应，也太真实了吧。
```

```text
[Uhm] 我再想一下，这个方案还可以再收一收。
```

## 3. 典型使用方式

下面示例默认你已经完成类似如下初始化：

```python
from streamvox import TTSEngine


engine = TTSEngine(
    model="voxcpm2-gguf",
    license_key="YOUR_SDK_KEY",
    device="auto",
    verify_model_sha256=False,
)
```

### 3.1 纯文本 Voice Design

适合没有参考音频，只想靠文本设计声音的场景。

```python
chunks = engine.stream(
    text="欢迎来到 StreamVox，接下来我将为你介绍这一代系统的核心升级。",
    mode="text",
    control_text="沉稳的中年男性科技产品讲解员，声音偏低，语速平稳，表达清晰，带少量发布会旁白感。",
)
```

### 3.2 单参考音色克隆

适合“保留音色，再说新文本”的场景。

```python
prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="example/reference.wav",
    prompt_text="这是参考音频对应的转写文本。",
    persist=False,
)

chunks = engine.stream(
    text="您好，您的订单已经发货，预计明天下午送达。",
    role_name=prompt,
    mode="ref",
    control_text="speaking very fast, bright and full",
)
```

### 3.3 continuation 续写

适合“沿着已有音频继续往后说”的场景。

```python
prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="example/reference.wav",
    prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
    persist=False,
)

chunks = engine.stream(
    text="接下来我们继续讨论第三个问题，它其实和前面两个问题共享同一个根源。",
    role_name=prompt,
    mode="continuation",
)
```

### 3.4 `ref_continuation`

适合希望同时保留音色和续写上下文的场景，但要记住：当前公开入口里它仍然基于**同一条** Prompt 音频。

```python
prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="example/reference.wav",
    prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
    persist=False,
)

chunks = engine.stream(
    text="如果继续往下说，这个命题真正困难的部分，其实是我们如何定义自己和他人的边界。",
    role_name=prompt,
    mode="ref_continuation",
)
```

## 4. 参考与致谢

感谢 VoxCPM 2 原始团队在多语种语音生成、自然语言控制、续写能力和更强表现力语音建模方向上的持续研究与贡献。  
